"""
extractor.py — Extracts beliefs and vocabulary from conversation text.

Two extractors:
  EmbeddingBeliefExtractor  — embedding gate (Qwen3-Embedding-0.6B) + regex extraction
  VocabularyMonitor         — tracks words being introduced or defined (unchanged)

Architecture
────────────
Stage 1 — Intent Gate  (Qwen/Qwen3-Embedding-0.6B via transformers)
    The user message is embedded and compared via cosine similarity against a
    small set of pre-embedded anchor phrases that represent "this message
    contains a storable factual claim".  Messages below GATE_THRESHOLD are
    skipped entirely — no extraction attempted, near-zero cost per turn.

    The model is loaded once at init with the same DEVICE/DTYPE used by
    ttt_engine.py, kept in eval() / inference_mode throughout, and never
    touches the optimizer — it only reads.

Stage 2 — Extraction  (original regex BeliefExtractor)
    On messages that pass the gate, the existing regex extractor handles
    structured {subject, predicate, value} extraction.  This is still the
    right tool for pulling values out of matched patterns — the embedding
    model's job is purely deciding whether it's worth trying.

    Why keep regex for stage 2?  Because extraction requires producing
    *structured output* (subject / predicate / value triples) from arbitrary
    phrasing.  An embedding alone can't do that — it gives you a score, not
    a parse.  The correct upgrade path for stage 2 would be an LLM call, but
    that means either a second model load or a round-trip; for now, regex on
    a pre-filtered set of messages is a clean, fast solution.

Fallback
    If the embedding model fails to load (e.g. not yet downloaded, OOM),
    _embed_ok is False and all messages are passed straight through to the
    regex extractor — identical behaviour to the original file.

VRAM note
    Qwen3-Embedding-0.6B at bfloat16 uses ~1.2 GB VRAM.  It shares the
    CUDA device with the TTT model but does not share any parameters or
    optimizer state.  Keep it in eval() and use @torch.inference_mode() on
    every embed call so its activations are never retained.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

log = logging.getLogger(__name__)

# ── Match ttt_engine.py so both models land on the same device/dtype ──
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16

EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
EMBED_MAX_LEN  = 512      # more than enough for chat messages; saves memory
GATE_THRESHOLD = 0.32     # cosine similarity floor; tune 0.28–0.40

# Task instruction prepended to the user message (not to anchors).
# Qwen3-Embedding is instruction-tuned; providing a short task description
# improves retrieval accuracy for asymmetric query/document pairs.
_QUERY_INSTRUCT = (
    "Instruct: Determine whether this conversational message contains "
    "a factual claim about the speaker or the world that is worth storing "
    "as a persistent belief.\nQuery: "
)


# ─────────────────────────────────────────────────────────────
# ANCHOR PHRASES
# These represent the semantic cluster of "storable belief messages".
# Keep the list small and diverse — the gate is a coarse filter.
# Anchors are NOT wrapped in the query instruction (they are the
# "document" side of the asymmetric pair).
# ─────────────────────────────────────────────────────────────

_BELIEF_ANCHORS: List[str] = [
    # Identity
    "my name is Alex",
    "people call me by a nickname",
    # Age
    "I am twenty-five years old",
    # Location
    "I live in Seattle",
    "I am originally from Tokyo",
    "I am currently based in London",
    # Work / study
    "I work at a software company",
    "I am employed as a software engineer",
    "I study computer science at university",
    "I am a professional musician",
    # Preferences
    "I really love hiking and the outdoors",
    "I strongly dislike spicy food",
    "my favourite book is Dune",
    # Relationships / possessions
    "my dog is called Max",
    "my girlfriend's name is Sarah",
    "I own a motorbike",
    # Corrections
    "actually that is wrong, I meant Seattle not Portland",
    "no wait, I meant something different earlier",
    "I was mistaken, let me correct that",
    # World facts
    "the capital of France is Paris",
    "water boils at one hundred degrees Celsius",
]


# ─────────────────────────────────────────────────────────────
# POOLING  (Qwen3-Embedding uses last-token pooling, not mean)
# This is a decoder-based model; the final hidden state of the
# last real token carries the sequence representation.
# ─────────────────────────────────────────────────────────────

def _last_token_pool(last_hidden: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Extract the embedding of the last non-padding token per sequence.
    Handles both left-padding (padding_side='left') and right-padding.
    """
    # With left-padding every sequence's last token IS the final token.
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden[:, -1]
    # Right-padding: walk to the last real token per row.
    seq_lens = attention_mask.sum(dim=1) - 1
    batch    = last_hidden.shape[0]
    return last_hidden[torch.arange(batch, device=last_hidden.device), seq_lens]


# ─────────────────────────────────────────────────────────────
# SHARED UTILITIES (used by both extractors)
# ─────────────────────────────────────────────────────────────

_CORRECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^actually[,\s]",
        r"^no[,\s]",
        r"^wait[,\s]",
        r"^i meant[,\s]",
        r"^i mean[,\s]",
        r"^correction[,\s]",
        r"^that('s| is) wrong",
        r"^you('re| are) wrong",
        r"not .+, it('s| is)",
        r"^i was wrong",
    ]
]

_DEFINITION_PATTERNS = [
    r"([a-zA-Z]+) means? ([^.!?]{5,100})",
    r"([a-zA-Z]+) is (?:a |an |the )?(?:word|term|phrase|concept) (?:for|meaning|that means) ([^.!?]{5,100})",
    r"(?:the word|the term) ([a-zA-Z]+) means? ([^.!?]{5,100})",
    r"([a-zA-Z]+)[:\s]+(?:that's|that is|it's|it is) ([^.!?]{5,100})",
]

def _is_correction(text: str) -> bool:
    return any(r.search(text.strip()) for r in _CORRECTION_PATTERNS)

def _extract_definitions(text: str) -> List[Dict]:
    results = []
    for pattern in _DEFINITION_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            try:
                word = m.group(1).strip().lower()
                defn = m.group(2).strip()
                if len(word) >= 2 and len(defn) >= 5:
                    results.append({"word": word, "definition": defn})
            except IndexError:
                pass
    return results


# ─────────────────────────────────────────────────────────────
# EMBEDDING BELIEF EXTRACTOR
# ─────────────────────────────────────────────────────────────

class EmbeddingBeliefExtractor:
    """
    Drop-in replacement for BeliefExtractor.

    Adds a semantic gate in front of the regex extractor: only messages
    whose embedding is meaningfully similar to at least one "belief-shaped"
    anchor phrase are passed through to extraction.  Everything else is
    skipped, eliminating false positives on casual chit-chat.

    The output schema is identical to BeliefExtractor so companion.py,
    database.py, and the rest of the pipeline need zero changes.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID for the embedding model.
        Default: "Qwen/Qwen3-Embedding-0.6B"
    gate_threshold : float
        Cosine similarity floor (default 0.32).
        Raise to 0.38-0.40 to be more selective (fewer false positives).
        Lower to 0.25-0.28 to catch more edge cases (more false positives).
    """

    def __init__(
        self,
        model_id:       str   = EMBED_MODEL_ID,
        gate_threshold: float = GATE_THRESHOLD,
    ):
        self.gate_threshold = gate_threshold
        self._embed_ok      = False
        self._anchor_vecs:  Optional[torch.Tensor] = None  # (N, D) on DEVICE
        self._fallback      = BeliefExtractor()

        self._load_model(model_id)

    # ── Model loading ────────────────────────────────────────────────

    def _load_model(self, model_id: str):
        print(f"\n  Loading embedding model: {model_id}")
        print(f"  Device: {DEVICE} | Dtype: {DTYPE}")
        try:
            # padding_side='left' is required by Qwen3-Embedding for correct
            # last-token pooling when batching sequences of different lengths.
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                padding_side="left",
            )
            self._model = AutoModel.from_pretrained(
                model_id,
                dtype=DTYPE,
                device_map=DEVICE,
                trust_remote_code=True,
            )
            self._model.eval()

            vram = self._vram_usage()
            print(f"  Embedding model loaded | VRAM: {vram}")

            self._precompute_anchors()

        except Exception as exc:
            log.warning(
                "Embedding model failed to load (%s). "
                "Falling back to regex-only extraction.",
                exc,
            )
            self._embed_ok = False

    def _vram_usage(self) -> str:
        if torch.cuda.is_available():
            used  = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"{used:.2f}/{total:.2f} GB"
        return "CPU"

    # ── Anchor pre-computation ───────────────────────────────────────

    def _precompute_anchors(self):
        """
        Embed all anchor phrases once and store as a (N, D) tensor.
        Anchors are NOT wrapped in the query instruction — they are the
        document side of the asymmetric pair.
        """
        vecs = self._embed_batch(_BELIEF_ANCHORS, add_instruction=False)
        if vecs is None:
            log.warning("Anchor pre-computation failed.")
            return
        self._anchor_vecs = vecs   # (N, D), L2-normalised, on DEVICE
        self._embed_ok    = True
        log.debug("Pre-computed %d anchor embeddings.", len(_BELIEF_ANCHORS))

    # ── Core embedding ───────────────────────────────────────────────

    @torch.inference_mode()
    def _embed_batch(
        self,
        texts: List[str],
        add_instruction: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        Embed a list of strings. Returns (N, D) L2-normalised tensor on
        DEVICE, or None on failure.

        add_instruction=True  -> prepend _QUERY_INSTRUCT to each text
                                  (use for user messages / queries)
        add_instruction=False -> embed as-is
                                  (use for anchors / documents)
        """
        try:
            if add_instruction:
                texts = [_QUERY_INSTRUCT + t for t in texts]

            batch = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=EMBED_MAX_LEN,
                return_tensors="pt",
            ).to(DEVICE)

            outputs    = self._model(**batch)
            embeddings = _last_token_pool(
                outputs.last_hidden_state, batch["attention_mask"]
            )
            return F.normalize(embeddings, p=2, dim=1)

        except Exception as exc:
            log.debug("Embed batch failed: %s", exc)
            return None

    # ── Gate ────────────────────────────────────────────────────────

    def _gate(self, text: str) -> Tuple[bool, float]:
        """
        Returns (passes_gate, max_cosine_similarity).

        The user message is embedded with the query instruction prepended;
        anchors were embedded without it.  This asymmetric approach matches
        how Qwen3-Embedding is trained (instruction on the query side only).

        On embedding failure: returns (True, 1.0) so extraction still runs
        rather than silently dropping beliefs.
        """
        vec = self._embed_batch([text], add_instruction=True)
        if vec is None:
            return True, 1.0

        # vec: (1, D),  _anchor_vecs: (N, D)
        # dot product of two L2-normalised vectors == cosine similarity
        sims    = (vec @ self._anchor_vecs.T).squeeze(0)   # (N,)
        max_sim = sims.max().item()
        return max_sim >= self.gate_threshold, max_sim

    # ── Public API (identical signature to BeliefExtractor) ─────────

    def is_correction(self, text: str) -> bool:
        return _is_correction(text)

    def extract_beliefs(self, text: str) -> List[Dict]:
        """
        Extract structured beliefs from a user message.
        Returns list of {subject, predicate, value, is_correction, confidence}.
        """
        if not self._embed_ok:
            # Embedding model unavailable — pass through to regex directly
            return self._fallback.extract_beliefs(text)

        passes, sim = self._gate(text)
        log.debug("Gate: sim=%.3f passes=%s text=%r", sim, passes, text[:60])

        if not passes:
            return []

        # Gate passed — hand off to regex for structured extraction
        return self._fallback.extract_beliefs(text)

    def extract_definitions(self, text: str) -> List[Dict]:
        return _extract_definitions(text)

    def extract_all(self, text: str) -> Dict:
        return {
            "beliefs":       self.extract_beliefs(text),
            "definitions":   self.extract_definitions(text),
            "is_correction": self.is_correction(text),
        }


# ─────────────────────────────────────────────────────────────
# LEGACY REGEX BELIEF EXTRACTOR
# Still the extraction engine for stage 2.  Also exported
# directly so existing code that imports BeliefExtractor still works.
# ─────────────────────────────────────────────────────────────

class BeliefExtractor:
    """
    Rule-based extraction of beliefs from natural language.

    Looks for patterns like:
      "My name is X"        -> subject=user, predicate=name, value=X
      "I work at X"         -> subject=user, predicate=works_at, value=X
      "X is Y"              -> subject=X, predicate=is, value=Y
      "I live in X"         -> subject=user, predicate=lives_in, value=X
      "I like/love/hate X"  -> subject=user, predicate=likes/hates, value=X
      "Actually X is Y"     -> correction signal
      "No, X is Y"          -> correction signal
      "X means Y"           -> vocabulary definition
    """

    BELIEF_PATTERNS = [
        # Identity
        (r"my name is ([A-Z][a-zA-Z\-']+)",           "user", "name"),
        (r"i'?m? called ([A-Z][a-zA-Z\-']+)",         "user", "name"),
        (r"call me ([A-Z][a-zA-Z\-']+)",               "user", "name"),
        (r"people call me ([A-Z][a-zA-Z\-']+)",        "user", "name"),
        # Age
        (r"i'?m? (\d{1,3}) years? old",               "user", "age"),
        (r"my age is (\d{1,3})",                        "user", "age"),
        # Location
        (r"i live in ([A-Za-z\s,]+?)(?:\.|,|$)",      "user", "lives_in"),
        (r"i'?m? from ([A-Za-z\s,]+?)(?:\.|,|$)",     "user", "from"),
        (r"i'?m? in ([A-Za-z\s]+?) (?:right now|currently|at the moment)", "user", "currently_in"),
        # Work / study
        (r"i work (?:at|for) ([A-Za-z\s&]+?)(?:\.|,|$)",    "user", "works_at"),
        (r"i'?m? a(?:n)? ([a-zA-Z\s]+?) (?:by profession|for work|for a living)", "user", "job"),
        (r"i'?m? a(?:n)? ([a-zA-Z\s]+?)(?:\.|,|$)",          "user", "is_a"),
        (r"i study ([a-zA-Z\s]+?)(?:\.|,|$)",                "user", "studies"),
        (r"i go to ([A-Za-z\s]+?) (?:university|college|school)", "user", "attends"),
        (r"i(?:'?m| am) a(?:n)? ([a-zA-Z\s]+?) (?:programmer|developer|engineer|designer|writer|teacher|doctor|nurse|student)", "user", "is_a"),
        (r"i (?:do |work in |work with )?([a-zA-Z\s]+?) (?:programming|development|coding)", "user", "works_with"),
        # Preferences
        (r"i (?:really )?love ([a-zA-Z\s]+?)(?:\.|,|$)",     "user", "loves"),
        (r"i (?:really )?like ([a-zA-Z\s]+?)(?:\.|,|$)",     "user", "likes"),
        (r"my favou?rite ([a-zA-Z\s]+?) is ([a-zA-Z\s]+?)(?:\.|,|$)", None, None),
        (r"i (?:really )?hate ([a-zA-Z\s]+?)(?:\.|,|$)",     "user", "dislikes"),
        (r"i don'?t like ([a-zA-Z\s]+?)(?:\.|,|$)",          "user", "dislikes"),
        # Possessions / relationships
        (r"my (?:pet )?(?:dog|cat|bird|rabbit|fish) (?:is called|is named|'?s named?) ([A-Za-z]+)", "user_pet", "name"),
        (r"my (?:partner|wife|husband|girlfriend|boyfriend)'?s? name is ([A-Za-z]+)", "user_partner", "name"),
        (r"i have (?:a |an )?([a-zA-Z\s]+?)(?:\.|,|$)",      "user", "has"),
        # World facts
        (r"([a-zA-Z\s]{3,30}) is (?:actually |really )?([a-zA-Z\s0-9,]{2,60})(?:\.|,|$)", None, None),
        (r"the ([a-zA-Z\s]+?) (?:is|are|was|were) ([a-zA-Z\s0-9,]{2,60})(?:\.|,|$)", None, None),
    ]

    def __init__(self):
        self._correction_re = _CORRECTION_PATTERNS

    def is_correction(self, text: str) -> bool:
        return _is_correction(text)

    def extract_beliefs(self, text: str) -> List[Dict]:
        results    = []
        text_clean = text.strip()
        is_corr    = self.is_correction(text_clean)
        base_conf  = 0.55 if is_corr else 0.45

        for pattern, subject, predicate in self.BELIEF_PATTERNS:
            m = re.search(pattern, text_clean, re.IGNORECASE)
            if not m:
                continue

            # Special case: "my favourite X is Y"
            if "favou?rite" in pattern and subject is None:
                try:
                    results.append({
                        "subject":       "user",
                        "predicate":     f"favourite_{m.group(1).strip()}",
                        "value":         m.group(2).strip(),
                        "is_correction": is_corr,
                        "confidence":    base_conf,
                    })
                except IndexError:
                    pass
                continue

            # Generic "X is Y"
            if subject is None and predicate is None:
                try:
                    subj = m.group(1).strip().lower()
                    val  = m.group(2).strip()
                    # Skip pronouns, short subjects, and first-person
                    # possessive phrases that are already covered by a
                    # specific pattern above (e.g. "my name", "my age",
                    # "my favourite", "your name").  Without this, a
                    # message like "My name is Kilo" produces both
                    # {user, name, Kilo} from the identity pattern AND
                    # {my name, is, Kilo} from this generic one.
                    _GENERIC_SKIP = {
                        "it","he","she","they","we","that","this",
                        "there","here","what","which",
                        # first-person possessives handled by specific patterns
                        "my name","my age","my job","my pet","my dog",
                        "my cat","my partner","my wife","my husband",
                        "my girlfriend","my boyfriend","my favourite",
                        "my favorite",
                        # second-person (AI identity corrections)
                        "your name",
                    }
                    if len(subj) < 3 or subj in _GENERIC_SKIP:
                        continue
                    results.append({
                        "subject":       subj,
                        "predicate":     "is",
                        "value":         val,
                        "is_correction": is_corr,
                        "confidence":    base_conf - 0.05,
                    })
                except IndexError:
                    pass
                continue

            # Standard pattern
            try:
                value = m.group(1).strip()
                if len(value) >= 1:
                    results.append({
                        "subject":       subject,
                        "predicate":     predicate,
                        "value":         value,
                        "is_correction": is_corr,
                        "confidence":    base_conf,
                    })
            except IndexError:
                pass

        # Deduplicate: if a specific pattern and the generic "X is Y" pattern
        # both fired for the same (subject, predicate), keep only the first
        # (specific patterns always appear earlier in BELIEF_PATTERNS).
        seen:   set  = set()
        unique: list = []
        for b in results:
            key = (b["subject"], b["predicate"])
            if key not in seen:
                seen.add(key)
                unique.append(b)

        return unique

    def extract_definitions(self, text: str) -> List[Dict]:
        return _extract_definitions(text)

    def extract_all(self, text: str) -> Dict:
        return {
            "beliefs":       self.extract_beliefs(text),
            "definitions":   self.extract_definitions(text),
            "is_correction": self.is_correction(text),
        }


# ─────────────────────────────────────────────────────────────
# VOCABULARY MONITOR — unchanged from original
# ─────────────────────────────────────────────────────────────

class VocabularyMonitor:
    """
    Monitors the conversation for:
      1. Words being explicitly defined by the user
      2. Rare or domain-specific words worth tracking
      3. Proper nouns (names of people, places, things)
    """

    STOPWORDS = {
        "the","a","an","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","shall",
        "should","may","might","must","can","could","i","you","he",
        "she","it","we","they","me","him","her","us","them","my",
        "your","his","its","our","their","this","that","these","those",
        "and","but","or","nor","for","yet","so","at","by","in","of",
        "on","to","up","as","into","through","with","about","against",
        "between","out","off","over","under","then","once","here",
        "there","when","where","why","how","all","both","each","few",
        "more","most","other","some","such","no","not","only","own",
        "same","than","too","very","just","from","what","which","who",
        "whom","well","also","back","even","still","way","go","get",
        "make","know","think","take","see","come","want","look","use",
        "find","give","tell","work","call","try","ask","need","feel",
        "become","leave","put","mean","keep","let","begin","show",
        "hear","play","run","move","live","believe","hold","bring",
        "happen","write","provide","sit","stand","lose","pay","meet",
        "include","continue","set","learn","change","lead","understand",
        "watch","follow","stop","create","speak","read","spend","grow",
        "open","walk","win","offer","remember","love","consider","appear",
        "buy","wait","serve","die","send","expect","build","stay","fall",
        "cut","reach","kill","remain","suggest","raise","pass","sell",
        "require","report","decide","pull","okay","ok","yeah","yes","no",
        "hey","hi","hello","bye","goodbye","please","thank","thanks",
        "sorry","excuse","pardon","sure","maybe","perhaps","probably",
        "really","actually","basically","generally","usually","often",
        "always","never","sometimes","already","again","still","yet",
    }

    MIN_WORD_LEN = 4

    def extract_notable_words(self, text: str, known_words: set) -> List[Dict]:
        results = []
        words   = re.findall(r"\b([a-zA-Z']{4,})\b", text)

        for word in words:
            w = word.lower().strip("'")
            if w in self.STOPWORDS or len(w) < self.MIN_WORD_LEN:
                continue
            if w in known_words:
                continue

            idx     = text.lower().find(w)
            start   = max(0, idx - 40)
            end     = min(len(text), idx + len(w) + 40)
            context = text[start:end].strip()

            reason = "general"
            if word[0].isupper() and not text.startswith(word):
                reason = "proper_noun"
            if len(w) > 10 or "-" in word or "_" in word:
                reason = "technical"

            results.append({"word": w, "context": context, "reason": reason})

        seen   = set()
        unique = []
        for r in results:
            if r["word"] not in seen:
                seen.add(r["word"])
                unique.append(r)

        return unique[:15]
