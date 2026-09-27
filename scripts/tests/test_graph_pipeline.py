"""Tests for the graph export + layout pipeline (scripts/utils/export_cosmograph.py, compute_layout.py).
Builds a tiny throwaway SQLite database; never touches processed_data/.
Run: python3 scripts/tests/test_graph_pipeline.py"""
import csv
import hashlib
import io
import math
import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "utils"))

import compute_layout  # noqa: E402
import export_cosmograph  # noqa: E402


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def centroid(points):
    return tuple(sum(p[k] for p in points) / len(points) for k in range(3))


def max_angle(tri):
    """Largest interior angle of a triangle, in degrees (180 = collinear)."""
    out = []
    for k in range(3):
        u, v = sub(tri[(k + 1) % 3], tri[k]), sub(tri[(k + 2) % 3], tri[k])
        cos = sum(x * y for x, y in zip(u, v)) / (norm(u) * norm(v))
        out.append(math.degrees(math.acos(max(-1.0, min(1.0, cos)))))
    return max(out)


def synthetic_nodes():
    """Three platforms with the same group mix as the real export (1 big, 2 small)."""
    nodes = []
    spec = {
        "reddit": [("reddit_public_thread", 300), ("reddit_chat", 60), ("reddit_user", 50)],
        "instagram": [("instagram_user", 90), ("instagram_public_thread", 25), ("instagram_comment_thread", 2)],
        "claude": [("claude_chat", 10), ("claude_user", 2)],
    }
    for platform, groups in spec.items():
        for group, count in groups:
            for i in range(count):
                nodes.append({"id": f"{group}_{i}", "group": group, "size": str(1 + (i * 7) % 50)})
    return nodes


class LayoutTests(unittest.TestCase):
    def test_fibonacci_directions_never_collinear_for_small_counts(self):
        # The old i/(n-1) spacing put n=2 on opposite poles: centre + 2 satellites on one line.
        for n in (1, 2, 3):
            pts = compute_layout.fibonacci_3d(n, 1.0)
            self.assertEqual(len(pts), n)
            for p in pts:
                self.assertAlmostEqual(norm(p), 1.0, places=6)
        a, b = compute_layout.fibonacci_3d(2, 1.0)
        self.assertGreater(norm(cross(a, b)), 0.3, "two satellites must not be collinear with the centre")
        self.assertLess(abs(compute_layout.fibonacci_3d(1, 1.0)[0][1]), 1e-9, "a single satellite sits on the equator")

    def test_three_platforms_form_a_triangle_not_a_line(self):
        nodes = synthetic_nodes()
        pos = compute_layout.compute_galaxy_layout(nodes)
        self.assertEqual(len(pos), len(nodes))
        by_platform = {}
        for n in nodes:
            by_platform.setdefault(n["group"].split("_")[0], []).append(pos[n["id"]])
        c = [centroid(v) for v in by_platform.values()]
        # The old layout put all three on the y axis: a 180° "triangle".
        self.assertLess(max_angle(c), 160, "platform centroids are (nearly) collinear")

        # Axis spans are comparable: no axis is squashed to a thin line.
        spans = [max(p[k] for p in pos.values()) - min(p[k] for p in pos.values()) for k in range(3)]
        self.assertGreater(min(spans[0], spans[1]) / max(spans), 0.25, spans)

    def test_groups_within_a_platform_are_not_collinear(self):
        nodes = synthetic_nodes()
        pos = compute_layout.compute_galaxy_layout(nodes)
        groups = {}
        for n in nodes:
            if n["group"].startswith("reddit"):
                groups.setdefault(n["group"], []).append(pos[n["id"]])
        c = [centroid(v) for v in groups.values()]
        self.assertLess(max_angle(c), 160)

    def test_gaps_scale_with_cluster_size(self):
        nodes = synthetic_nodes()
        pos = compute_layout.compute_galaxy_layout(nodes)
        pts = list(pos.values())
        extent = max(norm(sub(p, centroid(pts))) for p in pts)
        reddit = [pos[n["id"]] for n in nodes if n["group"] == "reddit_public_thread"]
        cluster = max(norm(sub(p, centroid(reddit))) for p in reddit)
        # The old fixed 1200-unit platform gap made the biggest cluster ~5% of the scene.
        self.assertGreater(cluster / extent, 0.12)

    def test_layout_is_deterministic_and_finite(self):
        nodes = synthetic_nodes()
        a = compute_layout.compute_galaxy_layout(nodes)
        compute_layout.random.random()  # global RNG state must not matter
        b = compute_layout.compute_galaxy_layout(nodes)
        self.assertEqual(a, b)
        self.assertTrue(all(math.isfinite(v) for p in a.values() for v in p))
        self.assertNotEqual(a, compute_layout.compute_galaxy_layout(nodes, seed=7))

    def test_empty_input(self):
        self.assertEqual(compute_layout.compute_galaxy_layout([]), {})


def make_db(path):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE Users (id INTEGER PRIMARY KEY AUTOINCREMENT, platform TEXT, raw_id TEXT, display_name TEXT,
                            UNIQUE(platform, raw_id));
        CREATE TABLE Threads (id INTEGER PRIMARY KEY AUTOINCREMENT, platform TEXT, platform_thread_id TEXT, title TEXT,
                              UNIQUE(platform, platform_thread_id));
        CREATE TABLE Messages (msg_id TEXT PRIMARY KEY, thread_id INTEGER, author_id INTEGER, timestamp_utc INTEGER,
                               content TEXT, parent_msg_id TEXT);
    """)
    conn.executemany("INSERT INTO Users VALUES (?,?,?,?)", [
        (1, "instagram", "17841", "Me Myself"),
        (2, "instagram", "99", "Friend, \"quoted\""),
        (3, "reddit", "spez", "spez"),
        (4, "reddit", "lurker", "lurker"),   # no messages
    ])
    long_title = "A very long public thread title that goes on and on, with commas\nand newlines " * 4
    conn.executemany("INSERT INTO Threads VALUES (?,?,?,?)", [
        (10, "instagram", "p1", long_title),
        (11, "instagram", "dm1", "DM with Friend"),
        (12, "reddit", "t3_x", "r/test (Thread x)"),
        (13, "reddit", "t3_empty", "no messages here"),
    ])
    conn.executemany("INSERT INTO Messages VALUES (?,?,?,?,?,?)", [
        ("m1", 10, 1, 1700000000, "hi", None),
        ("m2", 10, 2, 1700000500, "hey", None),
        ("m3", 10, 1, 1700009000, "bye", None),
        ("m4", 11, 1, 1690000000000, "ms timestamp", None),   # milliseconds
        ("m5", 12, 3, 1710000000, "post", None),
    ])
    conn.commit()
    conn.close()


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "memory.db"
        self.out = Path(self.tmp.name) / "graph"
        make_db(self.db)

    def export(self):
        with redirect_stderr(io.StringIO()):
            export_cosmograph.logging.disable(export_cosmograph.logging.INFO)
            try:
                counts = export_cosmograph.export_to_cosmograph(self.db, self.out, identity_map=("Naitik", {"Me Myself": "Naitik"}))
            finally:
                export_cosmograph.logging.disable(export_cosmograph.logging.NOTSET)
        with open(self.out / "cosmograph_nodes.csv", newline="", encoding="utf-8") as f:
            nodes = {r["id"]: r for r in csv.DictReader(f)}
        with open(self.out / "cosmograph_edges.csv", newline="", encoding="utf-8") as f:
            edges = list(csv.DictReader(f))
        return counts, nodes, edges

    def test_columns_dates_and_counts(self):
        (n_nodes, n_edges), nodes, edges = self.export()
        self.assertEqual((n_nodes, n_edges), (len(nodes), len(edges)))
        self.assertEqual(list(next(iter(nodes.values())).keys()), export_cosmograph.NODE_FIELDS)
        self.assertEqual(list(edges[0].keys()), export_cosmograph.EDGE_FIELDS)
        self.assertNotIn("T_13", nodes, "threads without messages are not exported")

        me = nodes["U_1"]
        self.assertEqual((me["label"], me["platform"], me["kind"]), ("Naitik", "instagram", "user"))
        self.assertEqual(me["messages"], "3")
        self.assertEqual(me["first_ts"], "1690000000", "millisecond timestamps are scaled to seconds")
        self.assertEqual(me["last_ts"], "1700009000")

        t = nodes["T_10"]
        self.assertEqual((t["kind"], t["platform"], t["group"]), ("thread", "instagram", "instagram_public_thread"))
        self.assertEqual((t["first_ts"], t["last_ts"], t["messages"]), ("1700000000", "1700009000", "3"))
        self.assertLessEqual(len(t["label"]), export_cosmograph.LABEL_CHARS + 3)
        self.assertTrue(t["label"].endswith("..."))
        self.assertGreater(len(t["title"]), len(t["label"]), "full title is kept for the details panel")
        self.assertLessEqual(len(t["title"]), export_cosmograph.TITLE_CHARS)
        self.assertNotIn("\n", t["title"])
        self.assertNotIn(",", t["title"])

        e = next(e for e in edges if (e["source"], e["target"]) == ("U_1", "T_10"))
        self.assertEqual((e["weight"], e["first_ts"], e["last_ts"]), ("2", "1700000000", "1700009000"))

        lurker = nodes["U_4"]
        self.assertEqual((lurker["messages"], lurker["first_ts"], lurker["last_ts"]), ("0", "", ""))
        self.assertEqual(nodes["U_2"]["label"], "Friend quoted")

    def test_thread_members_without_messages_get_weight_zero_edges(self):
        conn = sqlite3.connect(self.db)
        conn.executescript("""
            CREATE TABLE ThreadMembers (thread_id INTEGER, user_id INTEGER, PRIMARY KEY (thread_id, user_id));
            INSERT INTO Users VALUES (5, 'discord', '4242', 'Dana');
            INSERT INTO ThreadMembers VALUES (11, 5), (11, 1), (13, 5);   -- (11,1) already has messages; 13 is empty
        """)
        conn.commit()
        conn.close()
        _, nodes, edges = self.export()
        dana = [(e["target"], e["weight"], e["first_ts"], e["last_ts"]) for e in edges if e["source"] == "U_5"]
        self.assertEqual(dana, [("T_11", "0", "1690000000", "1690000000")])
        self.assertEqual(sum((e["source"], e["target"]) == ("U_1", "T_11") for e in edges), 1)
        self.assertEqual((nodes["U_5"]["messages"], nodes["U_5"]["size"], nodes["U_5"]["first_ts"]), ("0", "1", "1690000000"))
        self.assertEqual(nodes["T_11"]["messages"], "1")

    def test_every_group_gets_a_platform_colour_not_white(self):
        _, nodes, _ = self.export()
        for n in nodes.values():
            self.assertRegex(n["color"], r"^#[0-9a-fA-F]{6}$")
            self.assertNotEqual(n["color"].lower(), "#ffffff", n["group"])
        # instagram_public_thread was missing from GROUP_COLORS and rendered white.
        self.assertEqual(export_cosmograph.group_color("instagram_public_thread"),
                         export_cosmograph.shade("#E1306C", export_cosmograph.THREAD_SHADE))
        self.assertEqual(export_cosmograph.group_color("reddit_user"), "#FF4500")
        self.assertEqual(export_cosmograph.group_color("mastodon_user"), export_cosmograph.GROUP_COLORS["group"])

    def test_database_is_opened_read_only(self):
        before = hashlib.md5(self.db.read_bytes()).hexdigest()
        self.export()
        self.assertEqual(hashlib.md5(self.db.read_bytes()).hexdigest(), before)

    def test_missing_database_is_an_error_not_a_new_file(self):
        missing = Path(self.tmp.name) / "nope.db"
        export_cosmograph.logging.disable(export_cosmograph.logging.INFO)
        self.addCleanup(export_cosmograph.logging.disable, export_cosmograph.logging.NOTSET)
        with self.assertRaises(FileNotFoundError):
            export_cosmograph.export_to_cosmograph(missing, self.out, identity_map=(None, {}))
        self.assertFalse(missing.exists())

    def test_export_then_layout_round_trip(self):
        self.export()
        nodes = compute_layout.read_csv(str(self.out / "cosmograph_nodes.csv"))
        pos = compute_layout.compute_galaxy_layout(nodes)
        self.assertEqual(set(pos), {n["id"] for n in nodes})
        self.assertTrue(all(math.isfinite(v) for p in pos.values() for v in p))

    def test_to_epoch(self):
        self.assertEqual(export_cosmograph.to_epoch(1700000000), 1700000000)
        self.assertEqual(export_cosmograph.to_epoch(1700000000123), 1700000000)
        self.assertEqual(export_cosmograph.to_epoch("1700000000"), 1700000000)
        self.assertIsNone(export_cosmograph.to_epoch(None))
        self.assertIsNone(export_cosmograph.to_epoch("garbage"))


class GraphPageTests(unittest.TestCase):
    """The page must read the exported columns and keep its fallbacks honest."""

    def test_page_uses_export_columns_and_fallbacks(self):
        html = Path(REPO_ROOT, "sarthink_graph.html").read_text(encoding="utf-8")
        for needle in ("layout_x", "layout_y", "layout_z", "first_ts", "last_ts", "resolvePositions",
                       "fallbackLayout", "/api/thread/", "no-store"):
            self.assertIn(needle, html)
        # Platform colours in the UI match the export.
        for platform, color in export_cosmograph.PLATFORM_COLORS.items():
            self.assertIn(f"{platform}: '{color}'", html)


if __name__ == "__main__":
    unittest.main()
