from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Character:
    """A known character profile."""

    character_id: str
    display_name: str


class CharacterStore:
    """SQLite-backed store for character profiles and face embeddings."""

    def __init__(self, db_path: str):
        self._db_path = db_path

    def migrate(self) -> None:
        """Create required tables if missing."""

        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS character (
                  character_id TEXT PRIMARY KEY,
                  display_name TEXT NOT NULL,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS face_embedding (
                  embedding_id TEXT PRIMARY KEY,
                  character_id TEXT NOT NULL,
                  model_name TEXT NOT NULL,
                  vector BLOB NOT NULL,
                  l2_norm REAL NOT NULL,
                  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(character_id) REFERENCES character(character_id)
                )
                """
            )

    def create_character(self, display_name: str) -> str:
        """Create a character profile and return its ID."""

        character_id = f"ch_{uuid.uuid4().hex[:12]}"
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO character(character_id, display_name) VALUES(?, ?)",
                (character_id, display_name),
            )
        return character_id

    def add_embedding(self, character_id: str, model_name: str, vector: np.ndarray) -> str:
        """Store a face embedding vector for a character."""

        embedding_id = f"emb_{uuid.uuid4().hex[:12]}"
        vec = np.asarray(vector, dtype=np.float32)
        l2_norm = float(np.linalg.norm(vec) + 1e-9)
        vec = vec / l2_norm
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO face_embedding(embedding_id, character_id, model_name, vector, l2_norm)
                VALUES(?, ?, ?, ?, ?)
                """,
                (embedding_id, character_id, model_name, vec.tobytes(), l2_norm),
            )
        return embedding_id

    def match(self, model_name: str, query_vector: np.ndarray) -> tuple[str, str, float] | None:
        """Return best-matching (character_id, display_name, similarity) for the given embedding."""

        q = np.asarray(query_vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT c.character_id, c.display_name, e.vector
                FROM face_embedding e
                JOIN character c ON c.character_id = e.character_id
                WHERE e.model_name = ?
                """,
                (model_name,),
            ).fetchall()

        best: tuple[str, str, float] | None = None
        for character_id, display_name, vec_blob in rows:
            v = np.frombuffer(vec_blob, dtype=np.float32)
            sim = float(np.dot(q, v))
            if best is None or sim > best[2]:
                best = (str(character_id), str(display_name), sim)
        return best

    def _conn(self) -> sqlite3.Connection:
        """Open a SQLite connection with sane defaults."""

        if self._db_path != ":memory:":
            p = Path(self._db_path)
            if p.parent != Path("."):
                p.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn
