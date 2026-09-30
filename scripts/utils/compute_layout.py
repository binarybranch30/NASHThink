#!/usr/bin/env python3
"""
compute_layout.py
=================
Pre-computes the 3D positions of the memory map and writes layout_x, layout_y, layout_z into
cosmograph_nodes.csv. Run once; the HTML renders the baked positions.

Default: community layout. People are grouped by who talks together (Louvain communities on the
people-conversation graph, the owner's own accounts left out), groups are packed on a flat disc and
each is laid out with a small force simulation (networkx). Deterministic (seed 42).

    python3 scripts/utils/compute_layout.py                    # community layout
    python3 scripts/utils/compute_layout.py --layout galaxy    # old platform clusters (Fibonacci spheres)
    python3 scripts/utils/compute_layout.py --igraph           # galaxy + igraph DrL refinement

Graphs over COMMUNITY_MAX_NODES nodes use the galaxy layout automatically.
"""

import csv
import math
import random
import os
import sys
import time
from pathlib import Path

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

NODES_CSV = str(REPO_ROOT / 'processed_data' / 'graph' / 'cosmograph_nodes.csv')
EDGES_CSV = str(REPO_ROOT / 'processed_data' / 'graph' / 'cosmograph_edges.csv')

LAYOUT_SEED = 42   # deterministic jitter
random.seed(LAYOUT_SEED)

# ─── Platform topology ────────────────────────────────────────────────────────
# Master sphere radius — platform centroids sit on this sphere in 3D space
MASTER_R = 2200
PLATFORM_ORDER = ['twitter', 'reddit', 'instagram', 'discord', 'facebook',
                  'whatsapp', 'google', 'chatgpt', 'claude']

# Platform sphere scale (relative to sqrt-count formula)
PLATFORM_SCALE = {
    'twitter':   1.0,
    'reddit':    0.72,
    'instagram': 0.35,
    'discord':   0.22,
    'facebook':  0.22,
    'whatsapp':  0.35,
    'google':    0.35,
    'chatgpt':   0.22,
    'claude':    0.22,
}

# Within each platform, group types map to a radial fraction of the cluster.
# type_factor=1.0 → outermost ring; 0.3 → innermost (near hub threads)
TYPE_RADIAL = {
    'dm_group':        0.28,
    'comment_thread':  0.40,
    'chat':            0.46,
    'public_thread':   0.55,
    'user':            1.00,
}

SPHERE_SQRT_SCALE = 4.8   # cluster radius = sqrt(count) * this

# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_platform(group: str) -> str:
    for p in PLATFORM_ORDER:
        if group.startswith(p):
            return p
    return 'twitter'

def fibonacci_3d(n: int, radius: float) -> list[tuple]:
    """Distribute n points evenly on a sphere of given radius (3D Fibonacci).

    Uses the half-step offset (i + 0.5) / n: the classic i / (n - 1) form puts one point on a pole
    and two points on opposite poles, so two satellites always ended up collinear with their centre
    (three platforms rendered as one vertical line).
    """
    golden = math.pi * (3 - math.sqrt(5))
    pts = []
    for i in range(n):
        y = 1.0 - ((i + 0.5) / n) * 2.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        t = golden * i
        pts.append((math.cos(t) * r * radius, y * radius, math.sin(t) * r * radius))
    return pts

def get_type_factor(group: str) -> float:
    for suffix, factor in TYPE_RADIAL.items():
        if group.endswith(suffix):
            return factor
    return 0.70

def read_csv(path: str) -> list[dict]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

# ─── Fibonacci sphere ─────────────────────────────────────────────────────────

GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))

def fibonacci_sphere_positions(count: int, radius: float, cx=0.0, cy=0.0, cz=0.0, z_scale=0.5):
    """Distribute `count` points evenly on a sphere of `radius` centred at (cx,cy,cz)."""
    out = []
    for i in range(count):
        t = i / max(count - 1, 1)
        y = 1.0 - t * 2.0              # [-1, 1]
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = GOLDEN_ANGLE * i
        x = math.cos(theta) * r * radius + cx
        y2 = math.sin(theta) * r * radius + cy
        z = y * radius * z_scale + cz
        out.append((x, y2, z))
    return out

# ─── Galaxy layout ────────────────────────────────────────────────────────────

def compute_galaxy_layout(nodes: list[dict], seed: int = LAYOUT_SEED) -> dict[str, tuple]:
    """Hierarchical natural sphere layout.
    
    Platforms orbit the biggest platform. Subgroups orbit the biggest subgroup
    within their platform. Nodes form soft volume spheres without overlapping.
    Gaps scale with the cluster radii so small graphs don't become a few dots
    separated by empty space. Same input + seed -> same positions.
    """
    rng = random.Random(seed)
    CLUSTER_K = 5.5
    GROUP_GAP_K = 0.35      # gap between subgroup spheres, relative to their radii
    PLATFORM_GAP_K = 0.45   # gap between platform spheres, relative to their radii
    MIN_GAP = 40

    def group_gap(r1, r2):
        return max(MIN_GAP, GROUP_GAP_K * (r1 + r2))

    # Bucket nodes
    platforms = {}
    for n in nodes:
        p = n['group'].split('_')[0]
        platforms.setdefault(p, {}).setdefault(n['group'], []).append(n)

    positions = {}
    
    # Pre-calculate radii
    p_data = [] # (p_name, p_radius, g_data, total_nodes)
    for p_name, groups_dict in platforms.items():
        g_data = []
        for g_name, gnodes in groups_dict.items():
            g_radius = math.sqrt(len(gnodes)) * CLUSTER_K
            g_data.append((g_name, g_radius, gnodes))
        
        g_data.sort(key=lambda x: len(x[2]), reverse=True)
        
        if len(g_data) == 1:
            p_radius = g_data[0][1]
        else:
            p_radius = g_data[0][1] + 2 * max(x[1] for x in g_data[1:]) + group_gap(g_data[0][1], max(x[1] for x in g_data[1:]))
            
        p_data.append((p_name, p_radius, g_data, sum(len(x[2]) for x in g_data)))

    p_data.sort(key=lambda x: x[3], reverse=True)
    if not p_data:
        return positions
    
    # Place platforms
    p_positions = {}
    p0_name, p0_r, _, _ = p_data[0]
    p_positions[p0_name] = (0.0, 0.0, 0.0)
    
    if len(p_data) > 1:
        sats = p_data[1:]
        dirs = fibonacci_3d(len(sats), 1.0)
        for i, (p_name, p_r, _, _) in enumerate(sats):
            dist = p0_r + p_r + max(MIN_GAP, PLATFORM_GAP_K * (p0_r + p_r))
            p_positions[p_name] = (dirs[i][0]*dist, dirs[i][1]*dist, dirs[i][2]*dist)
            
    # Place groups within platforms
    for p_name, _, g_data, _ in p_data:
        px, py, pz = p_positions[p_name]
        
        g_positions = {}
        g0_name, g0_r, g0_nodes = g_data[0]
        g_positions[g0_name] = (px, py, pz)
        
        if len(g_data) > 1:
            sats = g_data[1:]
            dirs = fibonacci_3d(len(sats), 1.0)
            for i, (g_name, g_r, g_nodes) in enumerate(sats):
                dist = g0_r + g_r + group_gap(g0_r, g_r)
                g_positions[g_name] = (px + dirs[i][0]*dist, py + dirs[i][1]*dist, pz + dirs[i][2]*dist)
                
        # Distribute nodes within each group
        for g_name, g_r, g_nodes in g_data:
            gx, gy, gz = g_positions[g_name]
            
            gnodes_sorted = sorted(g_nodes, key=lambda n: int(n.get('size') or 1), reverse=True)
            count = len(gnodes_sorted)
            
            for i, n in enumerate(gnodes_sorted):
                fraction = i / max(count - 1, 1)
                r = g_r * (fraction ** 0.55)
                
                cos_phi  = rng.uniform(-1.0, 1.0)
                sin_phi  = math.sqrt(max(0.0, 1.0 - cos_phi ** 2))
                theta    = rng.uniform(0.0, 2.0 * math.pi)

                jitter = g_r * 0.045
                x = gx + r * sin_phi * math.cos(theta) + rng.gauss(0, jitter)
                y = gy + r * sin_phi * math.sin(theta) + rng.gauss(0, jitter)
                z = gz + r * cos_phi                   + rng.gauss(0, jitter * 0.8)

                positions[n['id']] = (x, y, z)
                
    return positions

# ─── legacy bucket helper (kept for igraph path) ─────────────────────────────
def _bucket_by_platform(nodes):
    """Bucket nodes by platform → {group → [node]}
    (only needed for igraph refinement seed)."""

    by_platform: dict[str, dict[str, list]] = {}
    for n in nodes:
        p = get_platform(n['group'])
        by_platform.setdefault(p, {}).setdefault(n['group'], []).append(n)

    positions: dict[str, tuple] = {}

    # Place 5 platform centroids evenly on a 3D sphere
    platform_positions = fibonacci_3d(len(PLATFORM_ORDER), MASTER_R)

    for k, platform in enumerate(PLATFORM_ORDER):
        cx, cy, cz = platform_positions[k]

        groups = by_platform.get(platform, {})
        total = sum(len(v) for v in groups.values())
        if total == 0:
            continue

        # Platform sphere radius scales with node count
        platform_r = math.sqrt(total) * SPHERE_SQRT_SCALE * PLATFORM_SCALE.get(platform, 0.5)

        for group, gnodes in groups.items():
            type_fac = get_type_factor(group)
            ring_r = platform_r * type_fac

            # Bigger nodes (more activity) toward the center of the ring
            gnodes_sorted = sorted(gnodes, key=lambda n: int(n.get('size') or 1), reverse=True)

            jitter_xy = ring_r * 0.04
            jitter_z  = ring_r * 0.02

            pts = fibonacci_sphere_positions(
                len(gnodes_sorted), ring_r, cx, cy, cz, z_scale=0.45
            )
            for i, n in enumerate(gnodes_sorted):
                x, y, z = pts[i]
                x += random.gauss(0, jitter_xy)
                y += random.gauss(0, jitter_xy)
                z += random.gauss(0, jitter_z)
                positions[n['id']] = (x, y, z)

    return positions

# ─── Optional: igraph DrL refinement ──────────────────────────────────────────

def refine_with_igraph(nodes: list[dict], edges: list[dict],
                       seed_positions: dict[str, tuple]) -> dict[str, tuple]:
    """Run igraph DrL starting from galaxy seed positions.

    This refines positions based on actual graph topology so that nodes with
    many shared neighbours genuinely attract each other.  Much faster than
    running d3-force in the browser because igraph's C core runs the full
    simulation in <5 min for 80k nodes.
    """
    try:
        import igraph as ig
    except ImportError:
        print("  igraph not found — keeping galaxy layout.  Install with: pip install igraph")
        return seed_positions

    idx = {n['id']: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), directed=False)

    edge_list, seen = [], set()
    for e in edges:
        s, t = idx.get(e.get('source')), idx.get(e.get('target'))
        if s is None or t is None or s == t:
            continue
        key = (min(s, t), max(s, t))
        if key not in seen:
            seen.add(key)
            edge_list.append(key)
    g.add_edges(edge_list)

    # Normalise seed to roughly [-20, 20] (DrL internal scale)
    scale = 20.0 / max(MASTER_R * 2, 1)
    seed_2d = [(seed_positions[n['id']][0] * scale,
                seed_positions[n['id']][1] * scale) for n in nodes]

    print(f"  Running igraph DrL on {g.vcount()} nodes / {g.ecount()} edges…")
    t0 = time.time()
    layout = g.layout_drl(seed=seed_2d)
    print(f"  DrL finished in {time.time() - t0:.1f}s")

    # Rescale back
    xs = [p[0] for p in layout]
    ys = [p[1] for p in layout]
    mx, my = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span   = max(max(xs) - min(xs), max(ys) - min(ys), 1)
    target = MASTER_R * 2.2

    refined = {}
    for i, n in enumerate(nodes):
        x = (layout[i][0] - mx) / span * target
        y = (layout[i][1] - my) / span * target
        z = seed_positions[n['id']][2]   # keep platform z-layering
        refined[n['id']] = (x, y, z)
    return refined

# ─── Community layout (default) ──────────────────────────────────────────────
# Groups people by who actually talks together instead of by platform. The owner's own accounts are
# left out of the calculation: they are in almost every conversation, so they would pull everything
# into one hub and draw spokes across the whole map.

COMMUNITY_MAX_NODES = 20000   # bigger graphs fall back to the fast galaxy layout
FLATTEN_Z = 0.35              # the map reads as a gently tilted disc, not a cloud


def is_owner(n: dict) -> bool:
    if str(n.get('me', '')).strip() in ('1', 'true', 'True'):
        return True
    return n.get('kind') == 'user' and str(n.get('label', '')).endswith('(me)')


def compute_community_layout(nodes: list[dict], edges: list[dict], seed: int = LAYOUT_SEED) -> dict[str, tuple]:
    import networkx as nx

    by_id = {n['id']: n for n in nodes}
    owners = {n['id'] for n in nodes if is_owner(n)}
    G = nx.Graph()
    G.add_nodes_from(sorted(by_id))
    for e in edges:
        s, t = e['source'], e['target']
        if s in owners or t in owners or s not in by_id or t not in by_id:
            continue
        try:
            w = float(e.get('weight') or 0)
        except ValueError:
            w = 0.0
        G.add_edge(s, t, weight=0.5 + math.log1p(w))

    loose = sorted(v for v in G.nodes if G.degree(v) == 0 and v not in owners)
    core = G.subgraph([v for v in G.nodes if G.degree(v) > 0]).copy()
    comms = nx.community.louvain_communities(core, weight='weight', seed=seed) if core.number_of_nodes() else []
    comms = sorted((sorted(c) for c in comms), key=lambda c: (-len(c), c[0]))

    positions: dict[str, tuple] = {}
    unit = 60.0                                   # world units per sqrt(node) of community radius
    golden = math.pi * (3 - math.sqrt(5))
    placed_area = 0.0
    for i, members in enumerate(comms):
        r = unit * 0.75 * math.sqrt(len(members)) + unit * 0.6
        # Sunflower packing: biggest communities in the middle, the rest spiral outwards.
        d = 0.0 if i == 0 else 1.45 * math.sqrt(placed_area / math.pi) + r * 0.9
        placed_area += math.pi * r * r * 1.1
        cx, cy = d * math.cos(i * golden), d * math.sin(i * golden)
        cz = (random.Random(seed + i).random() - 0.5) * r * 0.6
        if len(members) == 1:
            local = {members[0]: (0.0, 0.0, 0.0)}
        else:
            sub = core.subgraph(members)
            local = nx.spring_layout(sub, dim=3, seed=seed, weight='weight', k=1.1 / math.sqrt(len(members)),
                                     iterations=80)
        for v, p in local.items():
            positions[v] = (cx + p[0] * r, cy + p[1] * r, cz + p[2] * r * FLATTEN_Z)

    # Outer ring: conversations nobody else took part in (notes, AI chats), grouped by platform.
    if positions:
        extent = max(math.hypot(p[0], p[1]) for p in positions.values())
    else:
        extent = unit * 4
    ring_r = extent * 1.08 + unit
    loose.sort(key=lambda v: (by_id[v].get('platform', ''), v))
    for k, v in enumerate(loose):
        a = 2 * math.pi * k / max(len(loose), 1)
        jitter = random.Random(hash(v) & 0xffff).random()
        rr = ring_r * (1 + 0.06 * (jitter - 0.5))
        positions[v] = (rr * math.cos(a), rr * math.sin(a), (jitter - 0.5) * unit * 2 * FLATTEN_Z)

    # The owner's own accounts sit at the centre of everything they link to.
    for o in sorted(owners):
        linked = [positions[e['target'] if e['source'] == o else e['source']] for e in edges
                  if o in (e['source'], e['target']) and (e['target'] if e['source'] == o else e['source']) in positions]
        if linked:
            positions[o] = tuple(sum(p[j] for p in linked) / len(linked) for j in range(3))
        else:
            positions[o] = (0.0, 0.0, 0.0)
    return positions


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    use_igraph = '--force' in sys.argv or '--igraph' in sys.argv
    galaxy = '--layout' in sys.argv and sys.argv[sys.argv.index('--layout') + 1:][:1] == ['galaxy']

    print(f"Loading {NODES_CSV} …")
    nodes = read_csv(NODES_CSV)
    print(f"  {len(nodes)} nodes")
    if not nodes:
        sys.exit("No nodes to lay out. Run scripts/utils/export_cosmograph.py first.")

    print(f"Loading {EDGES_CSV} …")
    edges = read_csv(EDGES_CSV)
    print(f"  {len(edges)} edges")

    # ── Step 1: layout ────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        import networkx  # noqa: F401  (the community layout needs it)
        have_nx = True
    except ImportError:
        have_nx = False
        print("  networkx is not installed for this Python; using the galaxy layout "
              "(run with .venv/bin/python for the community layout)")
    if have_nx and not galaxy and not use_igraph and len(nodes) <= COMMUNITY_MAX_NODES:
        print("Computing community layout (people grouped by who talks together)…")
        positions = compute_community_layout(nodes, edges)
    else:
        print("Computing galaxy layout (platform clusters)…")
        positions = compute_galaxy_layout(nodes)
    print(f"  Done in {time.time() - t0:.2f}s  ({len(positions)} nodes positioned)")

    # ── Step 2: optional igraph refinement (galaxy layout only) ─────────────
    if use_igraph:
        print("Refining with igraph DrL (physics-based)…")
        positions = refine_with_igraph(nodes, edges, positions)

    # ── Step 3: write positions back into nodes CSV ───────────────────────────
    print("Writing positions to CSV…")
    fieldnames = list(nodes[0].keys())
    for col in ('layout_x', 'layout_y', 'layout_z'):
        if col not in fieldnames:
            fieldnames.append(col)

    for n in nodes:
        pos = positions.get(n['id'], (0.0, 0.0, 0.0))
        n['layout_x'] = f"{pos[0]:.4f}"
        n['layout_y'] = f"{pos[1]:.4f}"
        n['layout_z'] = f"{pos[2]:.4f}"

    write_csv(NODES_CSV, nodes, fieldnames)

    print(f"\n✓ Layout complete.  Positions written to:\n  {NODES_CSV}")
    print("\nNow start the UI:  HF_HUB_OFFLINE=1 .venv/bin/python scripts/api/server.py  → http://127.0.0.1:8000/")

if __name__ == '__main__':
    main()
