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

Every word here is a lowercase ASCII token, so it is safe to put into a LanceDB filter expression.
"""
import re

WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)

# Spelling variant -> canonical form. Only whole words are mapped; English words are never touched.
_VARIANT_GROUPS = {
    "nahi": ["nahin", "nhi", "nai", "nahii", "nahee", "nhin"],
    "bahut": ["bohot", "bhot", "bahot", "bohat", "bhut", "boht"],
    "kya": ["kia", "kyaa"],
    "kyun": ["kyu", "kyon", "kiu", "kyoon"],
    "accha": ["acha", "achha", "achcha", "acchha"],
    "yaar": ["yar", "yaara"],
    "mujhe": ["mujhey", "muje", "mujhko"],
    "main": ["mai"],
    "mein": ["mei"],
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
""".split())

# Short Hinglish words that are also English words ("main reason", "a new tab", "yoga mat"): treated as filler
# or negation only when the text is recognisably Hinglish (see looks_hinglish).
AMBIGUOUS = {"main", "par", "pe", "bas", "han", "tab", "jab", "din", "kal", "ab", "re", "ye", "wo", "hu", "tu",
             "sab", "ek", "mat", "na", "hum", "aaj", "toh", "ho", "ka", "ki", "ke", "ko", "se", "the", "bhi", "kar"}

# Negation words: ignored for topic matching only. English ones are listed for quote safety (safe_clip).
NEGATIONS_HI = {"nahi", "na", "mat", "nahin"}
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
    # Generic words: equivalent for evidence, never used to pull extra chats into results.
    "night":    {"expand": False, "en": ["night", "nights", "tonight"], "hi": ["raat"]},
    "friend":   {"expand": False, "en": ["friend", "friends"], "hi": ["dost"]},
    "home":     {"expand": False, "en": ["home"], "hi": ["ghar"]},
    "money":    {"expand": False, "en": ["money"], "hi": ["paisa"]},
    "love":     {"expand": False, "en": ["love"], "hi": ["pyaar"]},
    "happy":    {"expand": False, "en": ["happy", "happiness"], "hi": ["khush"]},
}

_EN_CONCEPT = {w: c for c, g in CONCEPTS.items() for w in g["en"]}
_HI_CONCEPT = {w: c for c, g in CONCEPTS.items() for w in g["hi"]}


def normalize(word):
    """Canonical spelling of one lowercase word (unchanged unless it is a known Hinglish variant)."""
    w = word.lower()
    return VARIANTS.get(w, w)


def looks_hinglish(word_list):
    """True when the words include an unambiguous Hinglish function word (hai, nahi, mujhe, yaar ...)."""
    return any((n := normalize(w)) in FILLER | NEGATIONS_HI and n not in AMBIGUOUS for w in word_list)


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

    `terms` are all surface forms searched for: the other language's words for every expandable concept in
    the query, plus (for Hinglish query words) all spelling variants, since embeddings miss those. English
    words already in the query are left to the normal vector search.
    """

    def __init__(self, query):
        self.concepts = []   # (concept, query word, language)
        terms = []
        for w in words(query):
            concept, lang = concept_of(w)
            if not concept or not CONCEPTS[concept]["expand"] or any(c == concept for c, _, _ in self.concepts):
                continue
            self.concepts.append((concept, w, lang))
            other = "hi" if lang == "en" else "en"
            terms += surface_forms(concept, other)
            if lang == "hi":
                terms += surface_forms(concept, "hi")
        self.terms = sorted(set(t for t in terms if re.fullmatch(r"[a-z]{3,}", t)))
        self.regex = re.compile(r"(?<![^\W_])(" + "|".join(map(re.escape, self.terms)) + r")(?![^\W_])", re.I) if self.terms else None

    def __bool__(self):
        return bool(self.terms)

    def filter_expression(self):
        """LanceDB prefilter: the chunk text contains one of the terms (substring; `found` checks whole words)."""
        return "(" + " OR ".join(f"lower(text) LIKE '%{t}%'" for t in self.terms) + ")"

    def found(self, text):
        """The expansion terms that occur in `text` as whole words (lowercase, first-seen order)."""
        if not self.regex:
            return []
        seen = []
        for m in self.regex.finditer(text or ""):
            t = m.group(1).lower()
            if t not in seen:
                seen.append(t)
        return seen
