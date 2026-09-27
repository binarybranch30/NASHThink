"""Read-only evaluation of Ask Sarthink on Hinglish, English and cross-language questions.

Runs a fixed question set against the running API (POST /api/ask; nothing is written anywhere) and prints,
per question: how many chunks were retrieved and judged relevant, how many of those relevant sources really
contain the topic (checked with a hand-written pattern per question, independent of the app's own matching),
the share of Hinglish sources, and the time taken. Only counts are printed, never archive text.

    python3 scripts/semantic/eval_hinglish.py                       # http://127.0.0.1:8000
    python3 scripts/semantic/eval_hinglish.py --url http://127.0.0.1:8765 --markdown
"""
import argparse
import json
import re
import sys
import time
import urllib.request

# (question, kind, pattern a truly relevant chunk's messages should match)
QUESTIONS = [
    ("mujhe neend nahi aati", "hinglish", r"neend|nind|sleep|slept|sleeping|so ?gaya|sona|soya"),
    ("exam ki tension", "hinglish", r"exams?|pariksha|imtihaa?n|papers?|tests?"),
    ("padhai kaisi chal rahi hai", "hinglish", r"padh\w*|parh\w*|stud(y|ying|ies|ied)"),
    ("ghar pe kya ladai hui", "hinglish", r"lad(a|aa)i|ladna|lad(e|a|i) |jhagd\w*|jhagad\w*|fight\w*|argu\w*"),
    ("paise ki problem", "hinglish", r"paisa|paise|pais(e|o)|pese|money|rupees?|rs"),
    ("jee ki tayyari", "hinglish", r"jee|mains|advanced"),
    ("mummy papa gussa", "hinglish", r"mumm(y|i)|papa|mom|dad|parents?|maa|gharwal\w*"),
    ("akela feel hota hai", "hinglish", r"akel(a|e|i)|alone|lonely|loneliness"),
    ("dost se baat nahi hui", "hinglish", r"dost\w*|friends?|yaar"),
    ("college ke baare mein stress", "mixed", r"college|clg"),
    ("I couldn't sleep", "english", r"sleep|slept|sleeping|neend|nind|insomnia"),
    ("when was I worried about exams", "english", r"exams?|pariksha|papers?"),
]
HINGLISH_MARKERS = set("hai h hain nahi nhi ni kya kyu mujhe mera meri yaar bhai kar raha rha tha thi bhi toh aur "
                       "hoga acha bol kaise abhi mai tu tum".split())
BODY_RE = re.compile(r"^(?:\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] [^:\n]{1,80}: ?|\[(?:POST|REPLY/COMMENT|TWEET|REPLY)\] .*$)", re.M)


def body(text):
    """Message text without timestamps/authors and without the thread title, so titles can't count."""
    return BODY_RE.sub(" ", text or "")


def is_hinglish(text):
    words = re.findall(r"[a-z]+", (text or "").lower())
    return sum(w in HINGLISH_MARKERS for w in words) >= 3


def ask(url, question):
    req = urllib.request.Request(url.rstrip("/") + "/api/ask", method="POST",
                                 data=json.dumps({"question": question, "limit": 20}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--markdown", action="store_true", help="print a Markdown table")
    args = ap.parse_args(argv)

    rows = []
    for q, kind, pattern in QUESTIONS:
        rx = re.compile(rf"(?<![^\W_])(?:{pattern})(?![^\W_])", re.I)
        t = time.perf_counter()
        a = ask(args.url, q)
        took = time.perf_counter() - t
        srcs = a["sources"]
        rel = [s for s in srcs if s["relevant"]]
        on_topic = sum(1 for s in rel if rx.search(body(s["text"])))
        rows.append({
            "question": q, "kind": kind, "confidence": a["confidence"],
            "relevant": a["evidence"]["relevant"], "considered": a["evidence"]["considered"],
            "shown_relevant": len(rel), "on_topic": on_topic,
            "precision": on_topic / len(rel) if rel else None,
            "hinglish_sources": sum(1 for s in srcs if is_hinglish(body(s["text"]))), "sources": len(srcs),
            "terms": a["evidence"]["terms"], "seconds": round(took, 1),
        })

    if args.markdown:
        print("| Question | Type | Confidence | Relevant / checked | On topic (of shown relevant) | Hinglish sources | Terms | Time |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in rows:
            prec = f"{r['on_topic']}/{r['shown_relevant']}" + (f" ({r['precision']:.0%})" if r["precision"] is not None else "")
            print(f"| {r['question']} | {r['kind']} | {r['confidence']} | {r['relevant']}/{r['considered']} | {prec} | "
                  f"{r['hinglish_sources']}/{r['sources']} | {', '.join(r['terms'])} | {r['seconds']} s |")
        shown = sum(r["shown_relevant"] for r in rows)
        print(f"\nOverall on-topic rate of relevant sources: {sum(r['on_topic'] for r in rows)}/{shown}"
              + (f" ({sum(r['on_topic'] for r in rows) / shown:.0%})" if shown else ""))
    else:
        json.dump(rows, sys.stdout, indent=1)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
