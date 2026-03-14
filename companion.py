"""
companion.py — Main companion AI orchestrator.

Combines:
  - TTT inner-loop weight adaptation (learns HOW the user communicates)
  - Belief store (learns WHAT the user tells it as facts)
  - Vocabulary module (learns NEW WORDS and their meanings)
  - Persistent memory across sessions (remembers everything)

The full pipeline per turn:
  1. Extract beliefs + vocabulary from user message → store them
  2. Build prompt with beliefs + vocabulary injected
  3. Generate response
  4. Run TTT inner-loop on user input + response (AFTER generation)
  5. Checkpoint weights to DB every N turns
"""

import re
import time
from typing import Dict, List, Optional

from database import (
    initialize_database,
    BeliefStore,
    VocabularyStore,
    ConversationLog,
    SessionStore,
)
from extractor import EmbeddingBeliefExtractor, VocabularyMonitor
from ttt_engine import TTTModelWrapper


# ─────────────────────────────────────────────────────────────
# RECOMMENDED MODELS (in order of preference for this goal)
# ─────────────────────────────────────────────────────────────
RECOMMENDED_MODELS = {
    "1": ("RetentionLabs/TTT-Linear-1.3B-Base-Books-32k",
          "Best — native TTT, 1.3B, 32k context, Books training"),
    "2": ("RetentionLabs/TTT-MLP-760M-Base-Pile-8k",
          "Good — native TTT-MLP, 760M, broad Pile training"),
    "3": ("RetentionLabs/TTT-Linear-760M-Base-Pile-8k",
          "Good — native TTT-Linear, 760M, fast inference"),
    "4": ("ibm-granite/granite-4.0-350m",
          "Retrofit — fast, light, good for testing"),
    "5": ("Qwen/Qwen3.5-0.8B",
          "Retrofit — modern architecture, 262K context"),
}

# Checkpoint every N turns
CHECKPOINT_INTERVAL = 3

# Minimum ASCII ratio for a response to be considered coherent enough
# to adapt on. Prevents TTT from learning from its own garbage output.
MIN_ASCII_RATIO = 0.85


# ─────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Builds a messages list in OpenAI/HuggingFace chat format:
      [{"role": "system", "content": ...},
       {"role": "user",   "content": ...},
       {"role": "assistant", "content": ...}, ...]

    This is passed to the tokenizer's apply_chat_template() so each
    model gets the exact special tokens it was trained on.
    """

    SYSTEM = """You are {ai_name}, an AI companion. Your name is {ai_name} — state it confidently when asked.

Personality: warm, curious, attentive. You remember what the user tells you and refer back to it naturally.

Hard rules — follow these exactly:
- Reply in 1-3 plain sentences. No exceptions.
- Do NOT use emojis, ever.
- Do NOT use markdown, bold, italics, bullet points, or HTML.
- Do NOT go off-topic or make things up.
- Do NOT ask multiple questions at once — one question maximum per reply.
- When the user tells you something about themselves, acknowledge it directly and remember it.
- When asked who the user is, recall what they have told you.
- If you don't know something, say so simply and move on."""

    def build(self,
              user_input:    str,
              beliefs_text:  str,
              vocab_text:    str,
              history:       List[Dict],
              pending_learn: List[str],
              ai_name:       str = "Companion") -> List[Dict]:
        """Returns a messages list, not a plain string."""

        # Build system content with injected knowledge
        system_text  = self.SYSTEM.format(ai_name=ai_name)
        system_parts = [system_text]

        if beliefs_text:
            system_parts.append(
                f"\n[Facts about the user — refer to these when relevant]\n{beliefs_text}"
            )
        if vocab_text:
            system_parts.append(f"\n[My vocabulary]\n{vocab_text}")
        if pending_learn:
            words = ", ".join(pending_learn[:5])
            system_parts.append(
                f"\n[Words I'm still learning: {words}]"
            )

        messages = [{"role": "system", "content": "\n".join(system_parts)}]

        # Add recent conversation history (skip the very last user entry —
        # that's the current turn, added below)
        recent = [e for e in history if e.get("content")][-20:]
        # Remove the last user message if it matches current input
        # to avoid duplication
        if recent and recent[-1]["role"] == "user" and \
                recent[-1]["content"] == user_input:
            recent = recent[:-1]

        # Keep history short — avoids KV cache overflow on long sessions
        for entry in recent[-6:]:
            role = entry["role"]
            messages.append({"role": role, "content": entry["content"]})

        # Current user turn
        messages.append({"role": "user", "content": user_input})

        return messages


# ─────────────────────────────────────────────────────────────
# COMPANION AI
# ─────────────────────────────────────────────────────────────

class CompanionAI:

    def __init__(self, user_id: str, model_id: str, ai_name: str = "Companion"):
        self.user_id  = user_id
        self.model_id = model_id
        self.ai_name  = ai_name
        self.turn     = 0

        # Initialize DB tables
        initialize_database()

        # Persistent stores
        self.beliefs  = BeliefStore(user_id)
        self.vocab    = VocabularyStore(user_id)
        self.log      = ConversationLog(user_id)
        self.session  = SessionStore(user_id)

        # Extraction tools
        self.extractor = EmbeddingBeliefExtractor()
        self.monitor   = VocabularyMonitor()

        # Prompt builder
        self.prompt_builder = PromptBuilder()

        # TTT model
        self.model = TTTModelWrapper(model_id)

        # Restore previous session
        self._load_session()

    # ── Session persistence ──────────────────────────────────────────

    def _load_session(self):
        data = self.session.load()
        if data:
            try:
                self.model.load_deltas(data["weight_deltas"])
                self.model.gate.load_state({
                    "ema":       data.get("gate_ema"),
                    "n_fired":   data.get("updates_fired", 0),
                    "n_skipped": data.get("updates_skipped", 0),
                })
                self.turn = data.get("total_turns", 0)
                print(f"  ✓ Session restored — {self.turn} prior turns")
            except Exception as e:
                print(f"  ⚠ Could not restore session weights: {e}")
        else:
            print("  ● Starting fresh session")

    def _save_session(self):
        gate  = self.model.gate
        blob  = self.model.serialize_deltas()
        self.session.save(
            weight_deltas_blob = blob,
            gate_ema           = gate.ema or 0.0,
            updates_fired      = gate.n_fired,
            updates_skipped    = gate.n_skipped,
            total_turns        = self.turn,
        )

    # ── Belief + vocabulary processing ──────────────────────────────

    def _process_input(self, text: str) -> Dict:
        """
        Extract and store beliefs and vocabulary from user input.
        Returns a summary of what was learned this turn.
        """
        learned = {
            "new_beliefs":     [],
            "reinforced":      [],
            "corrected":       [],
            "contested":       [],
            "new_words":       [],
            "definitions":     [],
            "is_correction":   False,
        }

        extraction = self.extractor.extract_all(text)
        learned["is_correction"] = extraction["is_correction"]

        # ── Process beliefs ──────────────────────────────────────────
        for belief in extraction["beliefs"]:
            source = "correction" if extraction["is_correction"] else "user"
            conf   = belief["confidence"] + (0.1 if extraction["is_correction"]
                                              else 0.0)
            result = self.beliefs.add_or_update(
                subject   = belief["subject"],
                predicate = belief["predicate"],
                value     = belief["value"],
                source    = source,
                initial_confidence = min(conf, 0.75),
            )

            action = result["action"]
            if action == "new":
                learned["new_beliefs"].append(
                    f"{belief['subject']} {belief['predicate']}: {belief['value']}"
                )
            elif action == "reinforced":
                learned["reinforced"].append(belief["value"])
            elif action == "corrected":
                learned["corrected"].append(
                    f"{result['old_value']} → {result['new_value']}"
                )
            elif action == "contested":
                learned["contested"].append(
                    f"still think: {result['held_value']} "
                    f"(challenger: {result['challenger']})"
                )

        # ── Process explicit definitions ─────────────────────────────
        for defn in extraction["definitions"]:
            result = self.vocab.encounter(
                word       = defn["word"],
                definition = defn["definition"],
                context    = text,
            )
            learned["definitions"].append(
                f"{defn['word']}: {defn['definition']}"
            )

        # ── Track notable words ──────────────────────────────────────
        acquired_words = {
            w["word"] for w in self.vocab.get_acquired(min_confidence=0.6)
        }
        notable = self.monitor.extract_notable_words(text, acquired_words)
        for w in notable:
            result = self.vocab.encounter(
                word    = w["word"],
                context = w["context"],
            )
            if result["status"] == "new":
                learned["new_words"].append(w["word"])

        return learned

    def _get_learning_words(self) -> List[str]:
        """Words the AI is still learning — to show in prompt."""
        return [w["word"] for w in self.vocab.get_unknown_words(
            min_confidence=0.5
        )[:6]]

    @staticmethod
    def _is_coherent(text: str) -> bool:
        """
        Returns True if the text looks like coherent output worth
        adapting on. Filters out garbage responses before they can
        poison the TTT weights.
        """
        if not text or len(text) < 5:
            return False
        ascii_ratio = sum(c.isascii() for c in text) / len(text)
        return ascii_ratio >= MIN_ASCII_RATIO

    # ── Main chat method ─────────────────────────────────────────────

    def chat(self, user_input: str, verbose: bool = False) -> str:
        self.turn += 1
        t_start    = time.time()

        # ── 1. Log user message ──────────────────────────────────────
        self.log.add("user", user_input)

        # ── 2. Extract beliefs + vocabulary from user message ────────
        learned = self._process_input(user_input)

        if verbose:
            self._print_learning_summary(learned)

        # ── 3. Build prompt with injected knowledge ───────────────────
        history      = self.log.get_recent(n=14)
        beliefs_text = self.beliefs.format_for_prompt()
        vocab_text   = self.vocab.format_for_prompt()
        pending      = self._get_learning_words()

        prompt = self.prompt_builder.build(
            user_input   = user_input,
            beliefs_text = beliefs_text,
            vocab_text   = vocab_text,
            history      = history,
            pending_learn= pending,
            ai_name      = self.ai_name,
        )

        # ── 4. Generate response ──────────────────────────────────────
        print("\nCompanion: ", end="", flush=True)
        response = self.model.generate(prompt)

        # ── 5. Log response ───────────────────────────────────────────
        self.log.add("assistant", response)

        # ── 6. TTT adaptation (AFTER generation, never before) ────────
        # Only adapt on user input. Adapting on the model's own response
        # risks poisoning TTT weights with imperfect outputs
        ttt_stats = self.model.adapt(user_input)

        if verbose:
            icon   = "✦" if ttt_stats["updated"] else "○"
            reason = ttt_stats["reason"]
            loss   = ttt_stats["loss"]
            ema    = ttt_stats.get("ema") or 0.0
            print(f"  [{icon} TTT | {reason} | loss={loss:.4f} | "
                  f"ema={ema:.4f} | "
                  f"efficiency={self.model.gate.efficiency:.1%} | "
                  f"vram={self.model.vram_usage()}]")

        # ── 7. Checkpoint ─────────────────────────────────────────────
        if self.turn % CHECKPOINT_INTERVAL == 0:
            self._save_session()

        if verbose:
            elapsed = time.time() - t_start
            print(f"  [turn {self.turn} | {elapsed:.1f}s total]")

        return response

    # ── Display helpers ───────────────────────────────────────────────

    def _print_learning_summary(self, learned: Dict):
        if learned["is_correction"]:
            print("  [Correction signal detected]")
        if learned["new_beliefs"]:
            for b in learned["new_beliefs"]:
                print(f"  [New belief: {b}]")
        if learned["reinforced"]:
            print(f"  [↑ Reinforced: {', '.join(learned['reinforced'][:3])}]")
        if learned["corrected"]:
            for c in learned["corrected"]:
                print(f"  [Corrected: {c}]")
        if learned["contested"]:
            for c in learned["contested"]:
                print(f"  [Contested: {c}]")
        if learned["new_words"]:
            print(f"  [New words encountered: {', '.join(learned['new_words'][:5])}]")
        if learned["definitions"]:
            for d in learned["definitions"]:
                print(f"  [Definition stored: {d}]")

    def stats(self):
        gate   = self.model.gate
        vstats = self.vocab.stats()
        beliefs = self.beliefs.get_all()

        print("\n═══ Companion Learning Stats ════════════════════════════════")
        print(f"  User ID:          {self.user_id}")
        print(f"  Model:            {self.model_id}")
        print(f"  Total turns:      {self.turn}")
        print(f"  Device / VRAM:    {self.model.vram_usage()}")
        print()
        print("  ── TTT Weight Adaptation ───────────────────────────────")
        print(f"  Updates fired:    {gate.n_fired}")
        print(f"  Updates skipped:  {gate.n_skipped}")
        print(f"  Gate efficiency:  {gate.efficiency:.1%}")
        ema = f"{gate.ema:.4f}" if gate.ema else "N/A"
        print(f"  EMA loss:         {ema}")
        print()
        print("  ── Belief Store ────────────────────────────────────────")
        print(f"  Total beliefs:    {len(beliefs)}")
        high = sum(1 for b in beliefs if b["confidence"] > 0.7)
        low  = sum(1 for b in beliefs if b["confidence"] < 0.4)
        corr = sum(1 for b in beliefs if b["is_corrected"])
        print(f"  High confidence:  {high}")
        print(f"  Low confidence:   {low}")
        print(f"  Corrected:        {corr}")
        print()
        print("  ── Vocabulary ──────────────────────────────────────────")
        print(f"  Total words:      {vstats['total']}")
        print(f"  Fully acquired:   {vstats['acquired']}")
        print(f"  Familiar:         {vstats['familiar']}")
        print(f"  Still learning:   {vstats['learning']}")
        avg = vstats["avg_confidence"]
        print(f"  Avg confidence:   {avg:.2f}" if avg else "  Avg confidence: N/A")
        print("═════════════════════════════════════════════════════════════\n")

    def show_beliefs(self):
        beliefs = self.beliefs.get_all()
        if not beliefs:
            print("  No beliefs stored yet.")
            return
        print(f"\n  ── Known Beliefs ({len(beliefs)}) ──")
        for b in beliefs[:25]:
            bar = "█" * int(b["confidence"] * 10) + \
                  "░" * (10 - int(b["confidence"] * 10))
            corr_flag = " [corrected]" if b["is_corrected"] else ""
            print(f"  [{bar}] {b['confidence']:.2f}  "
                  f"{b['subject']} {b['predicate']}: {b['value']}"
                  f"{corr_flag}")

    def show_vocabulary(self):
        acquired = self.vocab.get_acquired()
        learning = self.vocab.get_unknown_words()
        print(f"\n  ── Vocabulary ──")
        if acquired:
            print(f"  Acquired ({len(acquired)}):")
            for w in acquired[:15]:
                defn = f" — {w['definition']}" if w.get("definition") else ""
                print(f"    ✓ {w['word']} [{w['confidence']:.2f}]{defn}")
        if learning:
            print(f"  Still learning ({len(learning)}):")
            for w in learning[:10]:
                defn = f" — {w['definition']}" if w.get("definition") else "?"
                print(f"    ? {w['word']} [{w['confidence']:.2f}]{defn}")
