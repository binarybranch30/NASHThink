"""Small Hinglish (romanised Hindi + English) vocabulary shared by retrieval (search.py) and Ask (memory_brief.py).

The multilingual embedding model understands Hindi in Devanagari but treats romanised Hindi mostly as
unfamiliar tokens, so a Hinglish question matches Hinglish *style* (yaar, hai, nahi ...) rather than its topic,
and an English question rarely reaches Hinglish chats. This module supplies the missing pieces, deliberately
small and hand-curated so it stays predictable:

- FILLER: Hinglish function words that carry no topic ("hai", "mujhe", "yaar"); ignored for relevance matching.
- NEGATIONS: "nahi", "mat", ... also ignored for *topic* matching, but never removed from text: quotes keep
  them (see memory_brief.safe_clip) so an answer can't lose or reverse a negation.
- VARIANTS: common spelling variants mapped to one canonical form (nhi/nai -> nahi, bohot -> bahut ...).
- CONCEPTS: a limited set of topic words with their English and Hinglish forms (sleep <-> neend, study <->
  padhai ...). Groups marked expand=True are specific enough to drive retrieval expansion; the others
  (generic words like dost/friend, ghar/home) only count as equivalent when judging evidence.
- skeleton() / word_pattern(): romanised Hindi has no fixed spelling (ladai, ladaai, ladayi; tayyari, taiyari,
  tayari; padhai, padai), so any Hinglish word, listed here or not, is matched through a spelling-tolerant,
  whole-word pattern built from its letters. English words are never loosened this way.

Every word and pattern here is built from lowercase ASCII letters only, so it is safe to put into a LanceDB
filter expression.
"""
import re

WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)

# Spelling variant -> canonical form. Only whole words are mapped; English words are never touched.
_VARIANT_GROUPS = {
    # Chat shorthand. Single letters are only read as Hinglish in Hinglish text (see AMBIGUOUS).
    "hai": ["h", "hae", "haii"],
    "ke": ["k"],
    "nahi": ["nahin", "nhi", "nai", "nahii", "nahee", "nhin", "ni", "nii", "nhii"],
    "bahut": ["bohot", "bhot", "bahot", "bohat", "bhut", "boht"],
    "kya": ["kia", "kyaa"],
    "kyun": ["kyu", "kyon", "kiu", "kyoon"],
    "accha": ["acha", "achha", "achcha", "acchha"],
    "yaar": ["yar", "yaara"],
    "mujhe": ["mujhey", "muje", "mujhko"],
    "main": ["mai"],
    "mein": ["mei", "m", "me"],
    "kuch": ["kuchh", "kch"],
    "raha": ["rha"],
    "rahi": ["rhi"],
    "rahe": ["rhe"],
    "zyada": ["jyada", "zada", "jada", "ziada"],
    "padhai": ["padhaai", "padai", "parhai", "padhayi", "padhaayi", "padhaii"],
    "padhna": ["padhne", "padhta", "padhti", "padhte", "padhunga", "padhungi", "parhna", "padh"],
    "neend": ["nind", "neendh", "nindh", "neeend"],
    "pariksha": ["pareeksha", "parikshaa", "priksha", "parikcha"],
    "imtihaan": ["imtihan", "imtehan", "imtehaan"],
    "tension": ["tensn", "tenshun", "tensan"],
    "pareshan": ["pareshaan", "preshan", "pareshani", "pareshaani", "preshani"],
    "chinta": ["chintaa", "fikar", "fikr", "fikkar"],
    "thakan": ["thakaan", "thaka", "thaki", "thake", "thakk", "thakavat", "thakawat"],
    "bimar": ["beemar", "bimaar", "beemaar", "bimari", "beemari", "bimaari"],
    "bukhar": ["bukhaar", "bukhhar"],
    "naukri": ["naukari", "nokri", "nokari", "naukriyan"],
    "shaadi": ["shadi", "shaadiyan", "shaadee"],
    "gussa": ["gusse", "ghussa", "gussaa"],
    "dukhi": ["udaas", "udas", "udaasi", "udasi"],
    "dard": ["dardh"],
    "raat": ["raaton", "raato", "raatein"],
    "dost": ["dosto", "doston", "dostt"],
    "ghar": ["ghr"],
    "paisa": ["paise", "paisey", "pese", "paiso"],
    "pyaar": ["pyar", "pyaarr"],
    "khush": ["khushi", "khusi"],
    "ladai": ["ladaai", "ladayi", "laddai", "ladaiyan", "larai"],
    "jhagda": ["jhagada", "jhagra", "jhagde", "jhagdaa", "jhghda"],
    "tayyari": ["taiyari", "tayari", "taiyaari", "tayaari", "tyari", "tiyari"],
    "mummy": ["mumma", "mumy", "mummi", "mom"],
    "papa": ["pappa", "paapa"],
    "akela": ["akele", "akeli", "akelapan", "akelaa"],
    "rona": ["roya", "royi", "rota", "roti", "rote", "rone", "ro"],
    "darr": ["dar", "darta", "darti", "darte", "dara", "dari", "darna"],
    "dikkat": ["dikat", "dikkate", "diqqat"],
    # English chat abbreviations (they are normalised, never loosened).
    "college": ["clg", "colg"],
    "school": ["schl", "skl", "skool"],
}
VARIANTS = {v: canon for canon, vs in _VARIANT_GROUPS.items() for v in vs}

# Hinglish words ignored for topic matching (English stopwords live in memory_brief.STOPWORDS).
FILLER = set("""
hai hain ho hoon hu hun tha thi thay thhe ka ki ke ko se mein main mujhe mujh mera meri mere tera teri tere
tum tumhe tumko tumhara tumhari tu aap apna apni apne hum humne hame hamein maine mene usne unhe usse woh wo vo
ye yeh yaar bhai bro re kya kyun kaise kaisa kaisi kab kaha kahan kaun kitna kitni kitne bahut bhi toh aur par
pe ek koi kuch sab sabhi abhi ab phir fir wala wali wale raha rahi rahe karna karta karti karte kar kiya kiye
kari hua hui hue hoga hogi hota hoti hote lekin magar ya haan han accha matlab baare bare baat baatein aati aata
aate aaye aayi aana aa gaya gayi gaye lagta lagti lage laga lagi sakta sakti sakte chahiye wahi yahi bas sirf
jab tab agar kyunki isliye waise vaise thoda zyada din saal mahina hafte kal aaj parso kabhi hamesha
chal chalo chala chali jaa ja jao jaata jaati jaate gya gyi lena lene lo dena dene diya diye pata lag lagta dekh
dekho dekha bol bolo bola boli bata batao bataya fir abe arre haal liye tak baad pehle karu karun karega karenga
karo karne kiya kaisi hn hmm haa acha bc bkl mc bsdk pls plz ok okay lol bro
""".split())

# Short Hinglish words that are also English words ("main reason", "a new tab", "yoga mat"): treated as filler
# or negation only when the text is recognisably Hinglish (see looks_hinglish).
AMBIGUOUS = {"main", "par", "pe", "bas", "han", "tab", "jab", "din", "kal", "ab", "re", "ye", "wo", "hu", "tu",
             "sab", "ek", "mat", "na", "hum", "aaj", "toh", "ho", "ka", "ki", "ke", "ko", "se", "the", "bhi", "kar",
             "h", "k", "m", "me", "mein", "hai", "lo", "do", "ja", "bol", "fir", "pls", "ok", "okay", "lol", "bro",
             "hmm", "haa", "bc", "mc", "dekh", "chal", "lag", "tak", "lena", "haal", "kam"}

# English function words: a question with any of these and only ambiguous Hinglish words is English.
EN_FUNCTION = set("""
a an the is are was were be been am i you he she it we they my your his her its our their me him us them this
that these those of for to in on at by with from about as and or but if so do does did have has had not what when
where which who why how can could would should will
""".split())

# Negation words: ignored for topic matching only. English ones are listed for quote safety (safe_clip).
NEGATIONS_HI = {"nahi", "na", "mat", "nahin", "ni"}
NEGATIONS_EN = {"not", "no", "never", "nothing", "none", "nobody", "neither", "nor", "cannot", "can't", "cant",
                "don't", "dont", "didn't", "didnt", "doesn't", "doesnt", "isn't", "isnt", "wasn't", "wasnt",
                "won't", "wont", "wouldn't", "wouldnt", "couldn't", "couldnt", "shouldn't", "shouldnt",
                "haven't", "havent", "hasn't", "hasnt", "hadn't", "hadnt", "aren't", "arent", "weren't", "werent"}

# Topic concepts. English forms are listed in full (no stemming guesswork); Hinglish forms are canonical
# words whose spelling variants come from VARIANTS. Keep English sides to one word's inflections so English
# questions don't gain English synonyms.
CONCEPTS = {
    "sleep":    {"expand": True,  "en": ["sleep", "sleeping", "slept", "sleepy", "asleep", "sleepless", "insomnia"], "hi": ["neend"]},
    "study":    {"expand": True,  "en": ["study", "studying", "studies", "studied"], "hi": ["padhai", "padhna"]},
    "exam":     {"expand": True,  "en": ["exam", "exams", "examination", "examinations"], "hi": ["pariksha", "imtihaan"]},
    "stress":   {"expand": True,  "en": ["stress", "stressed", "stressing", "stressful"], "hi": ["tension"]},
    "worry":    {"expand": True,  "en": ["worry", "worried", "worrying", "worries"], "hi": ["chinta", "pareshan"]},
    "tired":    {"expand": True,  "en": ["tired", "tiredness"], "hi": ["thakan"]},
    "sick":     {"expand": True,  "en": ["sick", "sickness"], "hi": ["bimar"]},
    "fever":    {"expand": True,  "en": ["fever"], "hi": ["bukhar"]},
    "pain":     {"expand": True,  "en": ["pain", "painful"], "hi": ["dard"]},
    "job":      {"expand": True,  "en": ["job", "jobs"], "hi": ["naukri"]},
    "marriage": {"expand": True,  "en": ["marriage", "married", "marry"], "hi": ["shaadi"]},
    "angry":    {"expand": True,  "en": ["angry", "anger"], "hi": ["gussa"]},
    "sad":      {"expand": True,  "en": ["sad", "sadness"], "hi": ["dukhi"]},
    "fight":    {"expand": True,  "en": ["fight", "fights", "fighting", "fought", "argument", "arguments"], "hi": ["ladai", "jhagda"]},
    "parents":  {"expand": True,  "en": ["parents", "parent", "mother", "father", "dad"], "hi": ["mummy", "papa", "maa", "gharwale"]},
    "prep":     {"expand": True,  "en": ["preparation", "preparing", "prep"], "hi": ["tayyari"]},
    "cry":      {"expand": True,  "en": ["cry", "cried", "crying", "tears"], "hi": ["rona"]},
    "alone":    {"expand": True,  "en": ["alone", "lonely", "loneliness"], "hi": ["akela"]},
    "fear":     {"expand": True,  "en": ["fear", "scared", "afraid", "frightened"], "hi": ["darr"]},
    "money":    {"expand": True,  "en": ["money"], "hi": ["paisa"]},
    # Generic words: equivalent for evidence, never used to pull extra chats into results.
    "night":    {"expand": False, "en": ["night", "nights", "tonight"], "hi": ["raat"]},
    "friend":   {"expand": False, "en": ["friend", "friends"], "hi": ["dost"]},
    "home":     {"expand": False, "en": ["home"], "hi": ["ghar"]},
    "problem":  {"expand": False, "en": ["problem", "problems", "issue", "issues"], "hi": ["dikkat"]},
    "love":     {"expand": False, "en": ["love"], "hi": ["pyaar"]},
    "happy":    {"expand": False, "en": ["happy", "happiness"], "hi": ["khush"]},
}

_EN_CONCEPT = {w: c for c, g in CONCEPTS.items() for w in g["en"]}
_HI_CONCEPT = {w: c for c, g in CONCEPTS.items() for w in g["hi"]}


def normalize(word):
    """Canonical spelling of one lowercase word (unchanged unless it is a known Hinglish variant)."""
    w = word.lower()
    return VARIANTS.get(w, w)


# Hinglish topic words that are also everyday English ("tension"): they don't make a sentence Hinglish.
_ALSO_ENGLISH = {"tension", "mummy", "papa"}

# English question scaffolding that is never a topic, even inside a Hinglish question ("akela feel hota hai").
SCAFFOLD_EN = set("""
feel feeling feelings felt time times think thought remember talk talked tell told say said happen happened
happening something anything everything really actually
""".split())


def looks_hinglish(word_list):
    """True when the words are recognisably Hinglish: an unambiguous function word (hai, nahi, mujhe, yaar),
    a Hinglish topic word (neend, tayyari), or ambiguous short words ("jee ki tayyari") with no English
    function word ("the", "for", "is") alongside them."""
    norm = [normalize(w) for w in word_list]
    if any(n in FILLER | NEGATIONS_HI and n not in AMBIGUOUS for n in norm):
        return True
    if any(n in _HI_CONCEPT and n not in _ALSO_ENGLISH for n in norm):
        return True
    lowered = {w.lower() for w in word_list}
    return any(n in AMBIGUOUS and n in FILLER | NEGATIONS_HI for n in norm) and not (lowered & EN_FUNCTION)


_ASPIRATED = re.compile(r"([bdgjkpt])h")


def skeleton(word):
    """Spelling-independent form of a romanised Hindi word: ladaai/ladayi -> ladai, taiyari/tayyari -> tayari,
    padhaai -> padai, neendh -> nind. Only used to compare or loosen Hinglish words."""
    w = normalize(word)
    w = w.replace("aa", "a").replace("ee", "i").replace("oo", "u")
    w = w.replace("z", "j").replace("w", "v").replace("ph", "f").replace("q", "k")
    w = _ASPIRATED.sub(r"\1", w)
    w = w.replace("aiy", "ay")
    w = re.sub(r"(.)\1+", r"\1", w)
    return w.replace("ayi", "ai")


def word_pattern(word):
    """Regex source (letters, groups and quantifiers only; valid in Python and in LanceDB's regexp_like)
    matching the common spellings of one Hinglish word as a whole word's letters. Words shorter than four
    letters (jee, dil) are matched exactly: loosening them would hit unrelated words."""
    sk = skeleton(word)
    if not re.fullmatch(r"[a-z]+", sk):
        return None
    canon = normalize(word)
    listed = sorted({canon, word.lower(), *_VARIANT_GROUPS.get(canon, [])})   # clg, schl, nind ...
    listed = [w for w in listed if re.fullmatch(r"[a-z]+", w)]
    if len(sk) < 4:
        return "(?:" + "|".join(listed) + ")"
    out = []
    for k, ch in enumerate(sk):
        prev = sk[k - 1] if k else ""
        glide = "y?" if ch == "i" and prev == "a" else ""   # ladai / ladayi
        if ch in "aei" and k == len(sk) - 1 and len(sk) >= 5:
            out.append(glide + "[aei]+")                      # akela / akele / akeli
            continue
        if ch == "a":
            piece = "a+"
        elif ch == "i":
            piece = "(?:i+|e{2,})"
        elif ch == "u":
            piece = "(?:u+|o{2,})"
        elif ch in "eo":
            piece = ch + "+"
        elif ch == "y" and prev in "aeiou":
            piece = "i?y+"
        elif ch == "j":
            piece = "[jz]+h?"
        elif ch == "v":
            piece = "[vw]+"
        elif ch == "f":
            piece = "(?:f+|ph)"
        elif ch == "k":
            piece = "(?:k+|q)h?"
        elif ch in "bdgpt":
            piece = ch + "+h?"
        else:
            piece = ch + "+"
        out.append(glide + piece)
    return "(?:" + "|".join(["".join(out)] + listed) + ")"


def whole_word_regex(patterns):
    """Python regex for any of `patterns` as a whole word (case-insensitive)."""
    return re.compile(r"(?<![^\W_])(?:" + "|".join(patterns) + r")(?![^\W_])", re.I)


def is_filler(word, hinglish_context=True):
    """Hinglish filler or negation (both ignored for topic matching). Ambiguous short words only in Hinglish text."""
    w = normalize(word)
    if w in AMBIGUOUS and not hinglish_context:
        return False
    return w in FILLER or w in NEGATIONS_HI


def is_negation(word):
    """Any negation, English or Hinglish (for keeping quotes faithful; ambiguous words count, to be safe)."""
    w = word.lower()
    return normalize(w) in NEGATIONS_HI or w in NEGATIONS_EN


def concept_of(word):
    """(concept, 'en'|'hi') for a topic word in either language, else (None, None)."""
    w = normalize(word)
    if w in _EN_CONCEPT:
        return _EN_CONCEPT[w], "en"
    if w in _HI_CONCEPT:
        return _HI_CONCEPT[w], "hi"
    return None, None


def surface_forms(concept, lang):
    """Every spelling of a concept's words in one language (canonical forms plus their variants)."""
    g = CONCEPTS[concept]
    if lang == "en":
        return list(g["en"])
    out = []
    for canon in g["hi"]:
        out.append(canon)
        out.extend(_VARIANT_GROUPS.get(canon, []))
    return out


def words(text):
    return [w.lower() for w in WORD_RE.findall(text or "")]


class Expansion:
    """Retrieval expansion for one query: which topic words to look for, in which chats.

    `terms` are the curated surface forms searched for: the other language's words for every expandable
    concept in the query, plus (for Hinglish query words) all spelling variants, since embeddings miss those.
    English words already in the query are left to the normal vector search.

    `patterns` cover the rest of a Hinglish query: every other content word (ladai, jee, akela ...) as a
    spelling-tolerant whole-word pattern, so its chats are found even though it isn't in the vocabulary.
    Generic words (ghar, dost) are only used when the query has nothing more specific.
    """

    MAX_PATTERNS = 6

    def __init__(self, query):
        ws = words(query)
        self.concepts = []   # (concept, query word, language)
        self.patterns = []   # (canonical query word, regex source)
        terms = []
        for w in ws:
            concept, lang = concept_of(w)
            if not concept or not CONCEPTS[concept]["expand"] or any(c == concept for c, _, _ in self.concepts):
                continue
            self.concepts.append((concept, w, lang))
            other = "hi" if lang == "en" else "en"
            terms += surface_forms(concept, other)
            if lang == "hi":
                terms += surface_forms(concept, "hi")
        self.terms = sorted(set(t for t in terms if re.fullmatch(r"[a-z]{3,}", t)))

        if looks_hinglish(ws):
            specific, generic = [], []
            for w in ws:
                n = normalize(w)
                if (is_filler(w, True) or w in EN_FUNCTION or n in SCAFFOLD_EN or len(n) < 3 or not n.isascii()
                        or not n.isalpha()):
                    continue
                concept, _ = concept_of(n)
                if concept and CONCEPTS[concept]["expand"]:
                    continue   # already covered by the curated forms above
                bucket = generic if concept else specific
                if n not in bucket:
                    bucket.append(n)
            # Generic words (ghar, dost) only when nothing more specific is asked about.
            for n in (specific or ([] if self.concepts else generic))[:self.MAX_PATTERNS]:
                p = word_pattern(n)
                if p:
                    self.patterns.append((n, p))

        self.regex = whole_word_regex(map(re.escape, self.terms)) if self.terms else None
        self._pattern_res = [(n, whole_word_regex([p])) for n, p in self.patterns]

    def __bool__(self):
        return bool(self.terms or self.patterns)

    def filter_expression(self):
        """LanceDB prefilter: the chunk text contains one of the terms or patterns as a whole word. One
        regexp_like pass (about 0.25 s over 13k chunks) is ~10x faster than a LIKE per term, and exact."""
        alts = "|".join([re.escape(t) for t in self.terms] + [p for _, p in self.patterns])
        return f"(regexp_like(text, '(?i)(?:^|[^\\p{{L}}\\p{{N}}_])(?:{alts})(?:[^\\p{{L}}\\p{{N}}_]|$)'))"

    def found(self, text):
        """The expansion terms and pattern words that occur in `text` as whole words (first-seen order)."""
        seen = []
        if self.regex:
            for m in self.regex.finditer(text or ""):
                t = m.group(0).lower()
                if t not in seen:
                    seen.append(t)
        for n, rx in self._pattern_res:
            if n not in seen and rx.search(text or ""):
                seen.append(n)
        return seen
