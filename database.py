"""
database.py — Persistent storage for beliefs, vocabulary, and sessions.

Three tables:
  beliefs    — facts the AI has learned, with confidence scores
  vocabulary — words/concepts the AI has acquired
  sessions   — serialized TTT weight deltas per user
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple


DB_PATH = Path("./companion_data/companion.db")


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent access
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize_database():
    """Create all tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS beliefs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT    NOT NULL,
            subject         TEXT    NOT NULL,
            predicate       TEXT    NOT NULL,
            value           TEXT    NOT NULL,
            confidence      REAL    NOT NULL DEFAULT 0.5,
            reinforcements  INTEGER NOT NULL DEFAULT 1,
            contradictions  INTEGER NOT NULL DEFAULT 0,
            source          TEXT    NOT NULL DEFAULT 'user',
            created_at      REAL    NOT NULL,
            updated_at      REAL    NOT NULL,
            is_corrected    INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS vocabulary (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT    NOT NULL,
            word            TEXT    NOT NULL,
            definition      TEXT,
            example_context TEXT,
            confidence      REAL    NOT NULL DEFAULT 0.3,
            encounters      INTEGER NOT NULL DEFAULT 1,
            first_seen_at   REAL    NOT NULL,
            last_seen_at    REAL    NOT NULL,
            UNIQUE(user_id, word)
        );

        CREATE TABLE IF NOT EXISTS sessions (
            user_id         TEXT    PRIMARY KEY,
            weight_deltas   BLOB,
            gate_ema        REAL,
            updates_fired   INTEGER DEFAULT 0,
            updates_skipped INTEGER DEFAULT 0,
            total_turns     INTEGER DEFAULT 0,
            last_saved_at   REAL    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversation_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     TEXT    NOT NULL,
            role        TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            timestamp   REAL    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_beliefs_user
            ON beliefs(user_id, subject);
        CREATE INDEX IF NOT EXISTS idx_vocab_user
            ON vocabulary(user_id, word);
        CREATE INDEX IF NOT EXISTS idx_log_user
            ON conversation_log(user_id, timestamp);
    """)
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────
# BELIEF STORE
# ─────────────────────────────────────────────────────────────

class BeliefStore:
    """
    Tracks facts the AI has learned with confidence scores.

    Confidence model:
      0.0 - 0.3  : uncertain / weakly held
      0.3 - 0.6  : moderately confident
      0.6 - 0.85 : strongly held
      0.85 - 1.0 : near-certain (requires strong contradiction to update)

    A fact gains confidence with each reinforcement.
    A contradiction lowers confidence and may flip the value if
    the new source is more credible or repeated.
    """

    CONFIDENCE_GAIN = 0.12    # per reinforcement
    CONFIDENCE_DROP = 0.25    # per contradiction
    CERTAINTY_CAP   = 0.95    # max confidence
    MIN_CONFIDENCE  = 0.05    # floor

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.conn    = get_connection()

    def add_or_update(self, subject: str, predicate: str, value: str,
                      source: str = "user", initial_confidence: float = 0.45
                      ) -> Dict:
        """
        Add a new belief or reinforce an existing one.
        Returns the belief record and whether it was new.
        """
        now      = time.time()
        subject  = subject.lower().strip()
        predicate = predicate.lower().strip()

        existing = self._get(subject, predicate)

        if existing is None:
            # Brand new belief
            self.conn.execute("""
                INSERT INTO beliefs
                  (user_id, subject, predicate, value, confidence,
                   reinforcements, source, created_at, updated_at)
                VALUES (?,?,?,?,?,1,?,?,?)
            """, (self.user_id, subject, predicate, value,
                  initial_confidence, source, now, now))
            self.conn.commit()
            return {"action": "new", "confidence": initial_confidence,
                    "value": value}

        # Existing belief — same value or different?
        if existing["value"].lower() == value.lower():
            # Reinforcement — same fact again, confidence goes up
            new_conf = min(
                existing["confidence"] + self.CONFIDENCE_GAIN,
                self.CERTAINTY_CAP
            )
            self.conn.execute("""
                UPDATE beliefs SET
                  confidence     = ?,
                  reinforcements = reinforcements + 1,
                  updated_at     = ?
                WHERE id = ?
            """, (new_conf, now, existing["id"]))
            self.conn.commit()
            return {"action": "reinforced", "confidence": new_conf,
                    "value": value}
        else:
            # Contradiction — different value for same subject+predicate
            new_conf = max(
                existing["confidence"] - self.CONFIDENCE_DROP,
                self.MIN_CONFIDENCE
            )
            # If confidence drops below 0.3, flip to the new value
            if new_conf < 0.30:
                self.conn.execute("""
                    UPDATE beliefs SET
                      value          = ?,
                      confidence     = 0.40,
                      contradictions = contradictions + 1,
                      is_corrected   = 1,
                      source         = ?,
                      updated_at     = ?
                    WHERE id = ?
                """, (value, source, now, existing["id"]))
                self.conn.commit()
                return {"action": "corrected",
                        "old_value": existing["value"],
                        "new_value": value,
                        "confidence": 0.40}
            else:
                # Confidence shaken but not flipped yet — hold old belief
                self.conn.execute("""
                    UPDATE beliefs SET
                      confidence     = ?,
                      contradictions = contradictions + 1,
                      updated_at     = ?
                    WHERE id = ?
                """, (new_conf, now, existing["id"]))
                self.conn.commit()
                return {"action": "contested",
                        "held_value": existing["value"],
                        "challenger": value,
                        "confidence": new_conf}

    def get_all(self) -> List[Dict]:
        rows = self.conn.execute("""
            SELECT subject, predicate, value, confidence, reinforcements,
                   contradictions, is_corrected, source
            FROM beliefs WHERE user_id = ?
            ORDER BY confidence DESC, updated_at DESC
        """, (self.user_id,)).fetchall()
        return [dict(r) for r in rows]

    def search(self, query: str) -> List[Dict]:
        q = f"%{query.lower()}%"
        rows = self.conn.execute("""
            SELECT subject, predicate, value, confidence
            FROM beliefs
            WHERE user_id = ? AND (subject LIKE ? OR value LIKE ? OR predicate LIKE ?)
            ORDER BY confidence DESC
            LIMIT 10
        """, (self.user_id, q, q, q)).fetchall()
        return [dict(r) for r in rows]

    def get_confident_facts(self, min_confidence: float = 0.5) -> List[Dict]:
        rows = self.conn.execute("""
            SELECT subject, predicate, value, confidence
            FROM beliefs
            WHERE user_id = ? AND confidence >= ?
            ORDER BY confidence DESC
            LIMIT 20
        """, (self.user_id, min_confidence)).fetchall()
        return [dict(r) for r in rows]

    def _get(self, subject: str, predicate: str) -> Optional[Dict]:
        row = self.conn.execute("""
            SELECT * FROM beliefs
            WHERE user_id = ? AND subject = ? AND predicate = ?
        """, (self.user_id, subject, predicate)).fetchone()
        return dict(row) if row else None

    def format_for_prompt(self, limit: int = 15) -> str:
        facts = self.get_confident_facts(min_confidence=0.4)[:limit]
        if not facts:
            return ""
        lines = ["Things I know about this person and our world:"]
        for f in facts:
            conf = f["confidence"]
            certainty = "definitely" if conf > 0.8 else \
                        "probably" if conf > 0.6 else \
                        "think" if conf > 0.4 else "vaguely recall"
            lines.append(
                f"  - I {certainty} know: {f['subject']} {f['predicate']} {f['value']}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# VOCABULARY STORE
# ─────────────────────────────────────────────────────────────

class VocabularyStore:
    """
    Tracks words and concepts the AI is learning.

    Confidence model:
      < 0.3  : just encountered, meaning uncertain
      0.3-0.6: seen a few times, definition forming
      0.6-0.9: well-understood
      > 0.9  : fully acquired
    """

    CONFIDENCE_PER_ENCOUNTER = 0.15
    CERTAINTY_CAP            = 0.95

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.conn    = get_connection()

    def encounter(self, word: str, context: str = "",
                  definition: str = "") -> Dict:
        """
        Record an encounter with a word.
        Returns whether the word is new, familiar, or fully acquired.
        """
        now  = time.time()
        word = word.lower().strip()

        existing = self._get(word)

        if existing is None:
            conf = 0.20 if not definition else 0.35
            self.conn.execute("""
                INSERT INTO vocabulary
                  (user_id, word, definition, example_context,
                   confidence, encounters, first_seen_at, last_seen_at)
                VALUES (?,?,?,?,?,1,?,?)
            """, (self.user_id, word,
                  definition or None,
                  context[:500] if context else None,
                  conf, now, now))
            self.conn.commit()
            return {"status": "new", "word": word, "confidence": conf}

        new_conf = min(
            existing["confidence"] + self.CONFIDENCE_PER_ENCOUNTER,
            self.CERTAINTY_CAP
        )
        update_def = definition and (
            not existing["definition"] or existing["confidence"] < 0.5
        )
        self.conn.execute("""
            UPDATE vocabulary SET
              confidence      = ?,
              encounters      = encounters + 1,
              last_seen_at    = ?,
              definition      = COALESCE(?, definition),
              example_context = COALESCE(?, example_context)
            WHERE user_id = ? AND word = ?
        """, (new_conf,
              now,
              definition if update_def else None,
              context[:500] if context and not existing["example_context"] else None,
              self.user_id, word))
        self.conn.commit()

        status = "acquired" if new_conf >= 0.9 else \
                 "familiar"  if new_conf >= 0.5 else "learning"
        return {"status": status, "word": word, "confidence": new_conf}

    def get_unknown_words(self, min_confidence: float = 0.4) -> List[Dict]:
        """Words still being learned (low confidence)."""
        rows = self.conn.execute("""
            SELECT word, definition, confidence, encounters
            FROM vocabulary
            WHERE user_id = ? AND confidence < ?
            ORDER BY encounters DESC
            LIMIT 20
        """, (self.user_id, min_confidence)).fetchall()
        return [dict(r) for r in rows]

    def get_acquired(self, min_confidence: float = 0.7) -> List[Dict]:
        rows = self.conn.execute("""
            SELECT word, definition, confidence
            FROM vocabulary
            WHERE user_id = ? AND confidence >= ?
            ORDER BY confidence DESC
            LIMIT 30
        """, (self.user_id, min_confidence)).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> Dict:
        row = self.conn.execute("""
            SELECT
              COUNT(*) as total,
              SUM(CASE WHEN confidence >= 0.9 THEN 1 ELSE 0 END) as acquired,
              SUM(CASE WHEN confidence >= 0.5 AND confidence < 0.9 THEN 1 ELSE 0 END) as familiar,
              SUM(CASE WHEN confidence < 0.5 THEN 1 ELSE 0 END) as learning,
              AVG(confidence) as avg_confidence
            FROM vocabulary WHERE user_id = ?
        """, (self.user_id,)).fetchone()
        return dict(row)

    def format_for_prompt(self, limit: int = 10) -> str:
        acquired = self.get_acquired()[:limit]
        learning = self.get_unknown_words()[:5]
        lines = []
        if acquired:
            words = ", ".join(
                f"{w['word']}" +
                (f" ({w['definition']})" if w['definition'] else "")
                for w in acquired[:8]
            )
            lines.append(f"Words I know well: {words}")
        if learning:
            words = ", ".join(w['word'] for w in learning)
            lines.append(f"Words I'm still learning: {words}")
        return "\n".join(lines)

    def _get(self, word: str) -> Optional[Dict]:
        row = self.conn.execute("""
            SELECT * FROM vocabulary WHERE user_id = ? AND word = ?
        """, (self.user_id, word)).fetchone()
        return dict(row) if row else None


# ─────────────────────────────────────────────────────────────
# CONVERSATION LOG
# ─────────────────────────────────────────────────────────────

class ConversationLog:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.conn    = get_connection()

    def add(self, role: str, content: str):
        self.conn.execute("""
            INSERT INTO conversation_log (user_id, role, content, timestamp)
            VALUES (?,?,?,?)
        """, (self.user_id, role, content, time.time()))
        self.conn.commit()

    def get_recent(self, n: int = 20) -> List[Dict]:
        rows = self.conn.execute("""
            SELECT role, content, timestamp FROM conversation_log
            WHERE user_id = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (self.user_id, n)).fetchall()
        return list(reversed([dict(r) for r in rows]))

    def get_all(self) -> List[Dict]:
        rows = self.conn.execute("""
            SELECT role, content FROM conversation_log
            WHERE user_id = ? ORDER BY timestamp ASC
        """, (self.user_id,)).fetchall()
        return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────
# SESSION STORE (TTT weight deltas)
# ─────────────────────────────────────────────────────────────

class SessionStore:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.conn    = get_connection()

    def save(self, weight_deltas_blob: bytes, gate_ema: float,
             updates_fired: int, updates_skipped: int, total_turns: int):
        self.conn.execute("""
            INSERT OR REPLACE INTO sessions
              (user_id, weight_deltas, gate_ema, updates_fired,
               updates_skipped, total_turns, last_saved_at)
            VALUES (?,?,?,?,?,?,?)
        """, (self.user_id, weight_deltas_blob, gate_ema,
              updates_fired, updates_skipped, total_turns, time.time()))
        self.conn.commit()

    def load(self) -> Optional[Dict]:
        row = self.conn.execute("""
            SELECT * FROM sessions WHERE user_id = ?
        """, (self.user_id,)).fetchone()
        return dict(row) if row else None
