# Hinglish in Sarthink

About 81,000 of the archive's messages are Hinglish, meaning Hindi written in English letters and mixed
with English: 15–17% of Reddit, Instagram and Discord messages. Search, Ask and the written answers all
handle it on purpose. Everything below runs locally.

## The problem

- **The embedding model reads style, not topic.** It (`paraphrase-multilingual-MiniLM-L12-v2`) understands
  Devanagari Hindi, but it treats romanised Hindi mostly as unknown tokens. A Hinglish question therefore finds
  chats that *sound* Hinglish ("yaar", "hai", "nahi") rather than chats about the topic.
- **Spelling isn't fixed.** The same word appears as ladai / ladaai / ladayi, tayyari / taiyari / tayari, or
  neend / nind / neendh.
- **Chat shorthand hides meaning.** `h` = hai, `ni` = nahi (not), `m` = mein, `rha` = raha. Missing a
  negation reverses the meaning of a sentence.
- **Small local models misread it.** Hinglish also costs about twice as many tokens per character as English.

## How it is handled

**Vocabulary** (`scripts/semantic/hinglish.py`), small and hand-curated so it stays predictable:
- **Shorthand and spelling variants.** `h→hai`, `ni/nhi/nai→nahi`, `m/me→mein`, `k→ke`, `clg→college`,
  `schl→school`, and so on.
- **Filler.** Hinglish function words and chat fillers ("hai", "mujhe", "chal", "pata", "bc") never count as
  topics.
- **Negations.** They are never topics either, but they are never cut out of a quote.
- **Topics in both languages.** sleep ↔ neend, study ↔ padhai, exam ↔ pariksha, stress ↔ tension,
  fight ↔ ladai / jhagda, parents ↔ mummy / papa, preparation ↔ tayyari, alone ↔ akela, cry ↔ rona,
  fear ↔ darr, money ↔ paisa, …
- **Generic words.** ghar, dost and problem only count when nothing more specific is asked.

**Spelling tolerance.** `skeleton()` maps a romanised word to a spelling-independent form (ladaai / ladayi →
ladai, taiyari / tayyari → tayari). `word_pattern()` turns that into a whole-word regex that also accepts the
listed variants and the common inflections (akela / akele / akeli). Words of three letters or fewer, such as
"jee", are matched exactly. English words are never loosened.

**Retrieval** (`search.search_expanded`). The normal vector search runs as before. For a Hinglish question,
a second search uses the same query vector but is restricted to chunks containing the question's topic, as a
whole word, in any spelling or in the other language. This uses LanceDB's `regexp_like` and needs no extra
index: one pass over the 13k chunks takes about 0.25 s, about 10× faster than a `LIKE` per word. Its hits are
interleaved into the ranking. English questions without a listed topic are searched exactly as before.

**Evidence** (`scripts/api/memory_brief.py`). A chunk counts as evidence only if its messages mention the
question's topic:
- spelling variants, other-language words and shorthand all count;
- negations ("couldn't", "nahi") are never the topic;
- a generic word alone ("ghar") doesn't answer "ghar pe kya ladai hui";
- thread titles only match whole words, so "jee" no longer matches every chunk of r/JEENEETards.

For a long chunk, the part shown and sent on is the lines around the matches, not the chunk's first 2,000
characters. Before this change, the relevant lines of long Hinglish chats were often cut off.

**Written answers** (`scripts/api/answer_writer.py`):
- **Language.** Answers are always in English, and quotes keep the original Hinglish words.
- **Token budget.** The prompt is sized with the llama.cpp server's own tokenizer (`POST /tokenize`), because
  Hinglish takes about 2 characters per token against about 3.5 for English. Excerpts are shortened, then the
  lowest-ranked sources are dropped, until the prompt fits the profile's `prompt_tokens` (Best 2,000, Quick
  1,400). A Hinglish question used to build a 5,300-token prompt and take 27 minutes on Best.
- **Shorthand glossary.** When the sources are Hinglish, the model gets a short glossary (h = is, ni / nhi =
  not, …) and is told that a negation reverses the meaning.
- **Quotes are checked.** Every claim must carry a short exact quote. Each quote is checked against the source
  it cites, ignoring case and spelling variants. Quotes that aren't found are underlined in the UI and counted
  in the answer's note.
- **Heavy content.** When the evidence contains statements about wanting to die, suicide or self-harm, the
  model is told to report them plainly with an exact quote, never softened or reinterpreted. In testing, the 8B
  model had rewritten such a message as "worried about your parents' health".

## Measured effect

`scripts/semantic/eval_hinglish.py` is read-only and prints only counts. It asks 12 fixed questions through
`/api/ask`. For each one, it checks whether the sources Ask calls relevant really contain the topic, using a
hand-written pattern per question that is independent of the app's own matching.

| Question | Relevant before → after | On topic before → after |
| --- | --- | --- |
| mujhe neend nahi aati | 15 → 15 | 67% → 100% |
| exam ki tension | 25 → 25 | 60% → 65% (the rest mention *tension* only) |
| padhai kaisi chal rahi hai | 26 → 21 | 70% → 100% |
| ghar pe kya ladai hui | 2 → 16 | 0% → 100% |
| paise ki problem | 11 → 22 | 18% → 100% |
| jee ki tayyari | 22 → 22 | 10% → 95% |
| mummy papa gussa | 19 → 25 | 84% → 100% |
| akela feel hota hai | 0 → 19 | — → 95% |
| dost se baat nahi hui | 1 → 17 | 100% → 100% |
| college ke baare mein stress | 18 → 16 | 33% → 75% |
| I couldn't sleep | 23 → 23 | 95% → 95% |
| when was I worried about exams | 14 → 14 | 86% → 86% |
| **All** | **176 → 235** | **59% → 93%** |

"On topic" is measured over the top 20 sources returned per question. Measured on 2026-09-27 against the full
local index (13,326 chunks).

```bash
python3 scripts/semantic/eval_hinglish.py --markdown        # needs the API running on 127.0.0.1:8000
```

## Limits

- The vocabulary is small by design. A Hinglish word that isn't listed still gets spelling tolerance and whole-word
  search, but no English equivalent.
- Devanagari questions go through the embedding model only; about 1,000 messages are in Devanagari.
- The written answer comes from a 3B or 8B model. The glossary and quote checks make its mistakes much less
  likely and much easier to see, but they can't rule them out: check the cited sources.
