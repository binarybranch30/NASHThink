# Sarthink fictional demo archive

Aarav Mehta, 21 in 2026: final-year CSE student in Bengaluru, from Jaipur. Every person, company, school/community event and private conversation is fictional; public places and product names are used for setting. Contact addresses and handles are fixtures, not contact instructions.

## Run

```sh
python generate_demo_data.py
python generate_demo_data.py --output ./another_demo
python generate_demo_data.py --check-only
```

Python 3.9+; standard library only; no internet, model, dependency installation or input files. The default output is `./demo_archive/`. The script replaces its own known outputs without deleting unrelated files. A repeated run produces the same file contents, including deterministic IDs, MIME boundaries and ZIP entry timestamps.

## Import mapping

| Generated source | Sarthink destination |
|---|---|
| `chatgpt/` | `archive/chatgpt/` |
| `claude/` | `archive/claude/` |
| `whatsapp/` | `archive/whatsapp/` |
| `discord/` | `archive/discord/aaravdev/` |
| `instagram-aarav.frames.zip` | `archive/instagram-aarav.frames.zip` |
| `facebook-aaravmehta.zip` | `archive/facebook-aaravmehta.zip` |
| `reddit-export/` | `archive/reddit-export/` |
| `twitter/data/` | `archive/twitter/data/` |
| `google/Takeout/` | `archive/google/Takeout/` |
| `identity_map.json` | `config/identity_map.json` |
| `twitter_id_map.json` | `processed_data/metadata/twitter_id_map.json` |

**Keep evaluator sidecars outside ingestion:** `README_DEMO.md`, `contacts_map.json`, `demo_questions.md`, `evidence_index.json`, `manifest.json`, `validation_report.json` and this generator. Do not point a recursive text ingester at the entire deliverable ZIP. Copy only the mapped source folders/files. Importing the answer key would invalidate the no-information checks.

Other people’s accounts intentionally remain separate identities in current Sarthink. `identity_map.json` contains only Aarav’s aliases. `contacts_map.json` is ground truth for future linking, including two explicit do-not-merge pairs.

## Counts

| Platform | Records | Share | Threads |
|---|---:|---:|---:|
| whatsapp | 3840 | 32.0% | 21 |
| instagram | 1920 | 16.0% | 12 |
| discord | 1920 | 16.0% | 17 |
| reddit | 1020 | 8.5% | 31 |
| twitter | 888 | 7.4% | 29 |
| facebook | 636 | 5.3% | 6 |
| gmail | 120 | 1.0% | 33 |
| google_chat | 384 | 3.2% | 7 |
| youtube | 252 | 2.1% | 18 |
| chatgpt | 636 | 5.3% | 40 |
| claude | 384 | 3.2% | 25 |

Total: **12,000 message records**, including four abandoned ChatGPT branch records; **11,996 active records**. There are **239 conversations/threads** and 40 human contacts besides Aarav. Search/Gemini activity and profile metadata are supplementary and not included in that count. There are exactly 120 edge records (1%): media placeholders, emoji-only messages and WhatsApp system lines.

The explicit “about 120 emails” instruction takes precedence over the conflicting Gmail 7% target. Gmail is 1%; the remaining quotas are redistributed approximately in the original proportions. Quoted replies do not count as new email records.

Four contacts have at least five account families, eight have two or three, and twenty-eight have one. A Google account shared by Gmail/Chat counts as one account family. Counts below are authored records; owner/assistant/system records are separate. YouTube export comments are owner-only, so Devika’s YouTube alias is linked through her Instagram statement and owner comment mentions.

| Person | Relation | Account families | Authored records |
|---|---|---|---:|
| Rohan Iyer (p01) | best friend | whatsapp, instagram, discord, twitter, reddit, google | 905 |
| Meera Shah (p02) | design friend | whatsapp, instagram, discord, facebook, google | 597 |
| Kabir Rao (p03) | hackathon teammate | whatsapp, discord, twitter, reddit, google | 603 |
| Nisha Verma (p04) | flatmate | whatsapp, instagram, discord, facebook, google | 750 |
| Devika Sen (p05) | photography mentor | instagram, reddit, youtube | 51 |
| Arjun Menon (p06) | college senior | whatsapp, discord | 116 |
| Sana Ali (p07) | hackathon teammate | discord, google | 119 |
| Aditya Kulkarni (p08) | gym and running friend | whatsapp, instagram | 158 |
| Tara Bose (p09) | dog rescue volunteer | whatsapp, instagram | 71 |
| Vikram Sethi (p10) | former landlord | whatsapp, google | 42 |
| Leela Nair (p11) | photography friend | instagram, facebook | 101 |
| Omar Farooq (p12) | Rust collaborator | discord, twitter, reddit | 95 |
| Anjali Mehta (p13) | mother | whatsapp | 54 |
| Sanjay Mehta (p14) | father | whatsapp | 73 |
| Isha Mehta (p15) | sister | whatsapp | 58 |
| Savitri Mehta (p16) | grandmother | whatsapp | 55 |
| Manu Joshi (p17) | classmate | whatsapp | 44 |
| Priya Thomas (p18) | classmate | whatsapp | 62 |
| Suresh Yadav (p19) | lab partner | whatsapp | 52 |
| Rohan Kapoor (p20) | cricket friend; NOT Rohan Iyer | whatsapp | 42 |
| Kavya Nair (p21) | photo walk friend | instagram | 76 |
| Siddharth Das (p22) | photography mutual | instagram | 70 |
| Ananya Roy (p23) | illustration mutual | instagram | 30 |
| Ishan Bose (p24) | campus photographer | instagram | 70 |
| Pranav Kale (p25) | Discord regular | discord | 63 |
| Noor Qureshi (p26) | accessibility contributor | discord | 20 |
| Jules Martin (p27) | open-source contributor | discord | 62 |
| Hana Mori (p28) | game developer | discord | 50 |
| Kabir Sethi (p29) | Discord member; NOT Kabir Rao | discord | 37 |
| Elise Turner (p30) | photography mutual | reddit | 34 |
| Naveen Bhat (p31) | Bengaluru mutual | reddit | 20 |
| Lucas Meyer (p32) | programming mutual | reddit | 14 |
| Paolo Ricci (p33) | developer mutual | twitter | 8 |
| Aditi Sood (p34) | design mutual | twitter | 10 |
| Wei Lin (p35) | Rust mutual | twitter | 10 |
| Elena Cruz (p36) | indie developer | twitter | 8 |
| Rita D'Souza (p37) | community organiser | facebook | 23 |
| Harish Jain (p38) | Jaipur neighbour | facebook | 60 |
| Ananya Desai (p39) | Nimbus Labs recruiter | google | 6 |
| Mira Sinha (p40) | Kiteleaf recruiter | google | 3 |
| Aarav Mehta (me) | Owner | all platforms | 6762 |
| AI assistants | generated assistant turns | ChatGPT, Claude | 490 |
| System/digest | not extra human contacts | WhatsApp, Gmail | 26 |

Detailed per-person × per-platform authored counts, monthly activity and IST-hour counts are in `manifest.json`.

## Storyline × platform × month grid

| Storyline | Months | Platforms |
|---|---|---|
| Photography | Jan 2024 → Jun 2024 → Jan 2025 → Jun 2026 | WhatsApp / Instagram / Reddit / YouTube / Twitter |
| Sarthink | Feb / Jun / Aug / Sep 2026 | Discord / Google Chat / Twitter |
| Internship | Sep 2025 → Jan–May 2026 | Gmail / WhatsApp / ChatGPT / Claude |
| Stress and sleep | May 2024 → Nov 2025 → Feb 2026 | WhatsApp / Discord / ChatGPT |
| Goa | Nov–Dec 2024; later callbacks | WhatsApp / Instagram / Twitter DMs |
| Meera | Oct 2025 → Jan 2026 | Instagram / WhatsApp |
| Family | Mar–Apr 2025; Apr–Aug 2026 | WhatsApp / Gmail / Instagram |
| Fitness | Jan / May 2025; Jan / Aug 2026 | WhatsApp / Instagram |
| Biscuit | Sep 2025 onward | Instagram / WhatsApp |
| Flat move | Jul–Aug 2025 | WhatsApp / Gmail / Reddit |
| Rust | Aug 2024 → Jun 2025 → May–Jul 2026 | Twitter / Discord / Claude / Reddit |
| Reddit community | 2024–2026; Feb 2025 argument | Reddit / Instagram / Discord |

## Format decisions and limitations

- The pasted specification lost spans of JSON/CSV schemas. The generator reconstructs the missing identity-map object, contact alias container, Reddit headers, Google Chat timestamp and YouTube headers. It has not been run against Sarthink’s actual parsers because their source was not provided. These choices are explicit in `REDDIT_FIELDS`, `YOUTUBE_FIELDS`, `OWNER_ALIASES` and the writer functions.
- WhatsApp uses Android `M/D/YY, h:mm AM/PM - Name: text`, with US month/day ordering and IST (UTC+05:30). Continuation lines have no new prefix; empty system lines are retained. All other message times are UTC.
- Meta archives use real inbox path shapes, newest-first messages and UTF-8 bytes represented as Latin-1 with JSON escapes. Decode the string with `s.encode("latin-1").decode("utf-8")` after JSON parsing. Profile metadata is in `personal_information/personal_information.json`. Media messages retain metadata, not binary media.
- Discord is DiscordChatExporter-shaped data with both sides, increasing snowflake IDs, reply references and attachment metadata. It is not a Discord official own-messages-only package.
- Reddit public posts/comments and Twitter tweets contain only Aarav’s side; their DMs/chats contain both sides. Reddit chat uses the `message` column and `thread_parent_message_id`. Public thread replies point to exported parents. Twitter JavaScript prefixes are `window.YTD.account.part0`, `window.YTD.tweets.part0` and `window.YTD.direct_messages.part0`.
- Google Takeout YouTube columns are: `Channel ID`, `Comment ID`, `Comment create timestamp`, `Price`, `Parent comment ID`, `Post ID`, `Video ID`, `Comment text`. `Comment text` contains a JSON object with `text`. These are owner comments, not a scrape of everybody’s replies.
- Gmail is a valid UTF-8 MIME mbox with From/To/Delivered-To, RFC dates, message/thread references, labels, deterministic multipart alternatives and quoted replies. Company domains use `.example`; documentation IPs use `192.0.2.21`.
- ChatGPT has a null-message root per conversation and two genuine abandoned edit branches. Follow `current_node` back through parents to select the kept transcript. Claude contains 25 conversations with parent-linked messages.
- Human messages include authored scenes, connected small-talk exchanges and reusable mini-discussions. The latter are synthetic variations, not LLM-written unique biographies. Date sampling dips in May 2024 and November 2025, bursts around Goa and the hackathon, and favours evenings with some post-midnight chats. Hinglish is sampled at 20% for eligible Indian friend/family exchanges; authored dialogue adds additional Hinglish.
- The source corpus intentionally has no exact CGPA, blood group, Japan trip, camera serial number or full-time Nimbus salary. Future wedding plans are not treated as a completed event.

## Verification

`python generate_demo_data.py --check-only` reads the written exports back and checks counts, chronology, JSON/CSV/MIME validity, identity coverage, graph references, Meta decoding, canonical facts and evidence paths. Generation runs these checks automatically. `validation_report.json` records their scope; external Sarthink parser compatibility remains unverified.

