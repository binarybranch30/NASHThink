# NASH Think sample data

`sample_archive/` is a **fictional** archive for demos: Aarav Mehta, a final-year CSE student in Bengaluru, with about
12,000 messages across WhatsApp, Instagram, Facebook, Discord, Reddit, X/Twitter, Gmail, Google Chat, YouTube comments,
ChatGPT and Claude (January 2024 – September 2026). Every person, company and conversation in it is invented.

- `sample_archive/README_DEMO.md`: what is in it, counts, and where each export goes
- `sample_archive/demo_questions.md`: 30 questions with expected answers and their evidence (the answer key)
- `sample_archive/contacts_map.json`: which accounts on different apps belong to the same person (ground truth)

The answer key, the contacts map and the other evaluator files (`evidence_index.json`, `manifest.json`,
`validation_report.json`) must not be imported: copy only the export folders and files listed in `README_DEMO.md`.

## Build a sample workspace

Every script reads and writes data next to itself, so the sample gets its own folder with a copy of `scripts/`:

```bash
D=~/nashthink-demo; S=$PWD/demo/sample_archive
mkdir -p $D/config $D/archive/discord/aaravdev $D/archive/twitter $D/archive/google $D/processed_data/metadata $D/processed_data/semantic
rsync -a --exclude __pycache__ scripts $D/ && ln -s $PWD/.venv $D/.venv && mkdir -p $D/.cache && cp -r .cache/models $D/.cache/
cp -r $S/chatgpt $S/claude $S/whatsapp $S/reddit-export $D/archive/ && cp $S/discord/*.json $D/archive/discord/aaravdev/
cp $S/*.zip $D/archive/ && cp -r $S/twitter/data $D/archive/twitter/ && cp -r $S/google/Takeout $D/archive/google/
cp $S/identity_map.json $D/config/ && cp $S/twitter_id_map.json $D/processed_data/metadata/
cd $D
for p in chatgpt claude meta reddit twitter google; do .venv/bin/python scripts/parsers/${p}_parser.py; done
.venv/bin/python scripts/parsers/discord_parser.py --archive archive/discord/aaravdev
TZ=Asia/Kolkata .venv/bin/python scripts/parsers/whatsapp_parser.py      # the sample's WhatsApp times are IST
.venv/bin/python scripts/utils/export_cosmograph.py && .venv/bin/python scripts/utils/compute_layout.py
.venv/bin/python scripts/semantic/chunk_builder.py
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/python scripts/semantic/embedder.py \
  --input processed_data/semantic/session_chunks.json --topics-only --topics-table topics
```

Then point a workspace at it in the main checkout's `config/workspaces.json` (see the main README), or serve it alone.
Don't run `scripts/context/*` on the sample: its IDs are fake and would be sent to the real Twitter/Reddit APIs.
