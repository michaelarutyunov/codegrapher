"""Core data models and database schema for CodeGrapher.

This module defines Pydantic models for in-memory validation and SQLite
persistence for the code graph index.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


# Constants from PRD Section 5
EMBEDDING_DIM = 768  # jina-embeddings-v2-base-code output dimension
DEFAULT_TOKEN_BUDGET = 3500


class Symbol(BaseModel):
    """A code element (function, class, or variable) in the repository.

    Attributes:
        id: Fully qualified name (e.g., "mymodule.MyClass.method")
        file: Relative path from repo root
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (inclusive)
        signature: Function or class signature
        doc: First sentence of docstring, if present
        mutates: Comma-separated list of mutated variables
        embedding: 768-dim vector as float32 numpy array
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="Fully qualified symbol name")
    file: str = Field(..., description="Relative path from repo root")
    start_line: int = Field(..., ge=1, description="Starting line number")
    end_line: int = Field(..., ge=1, description="Ending line number")
    signature: str = Field(..., description="Function or class signature")
    doc: Optional[str] = Field(None, description="First sentence of docstring")
    mutates: str = Field("", description="Comma-separated mutated variables")
    embedding: np.ndarray = Field(
        ..., description="768-dim embedding vector"
    )

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: np.ndarray) -> np.ndarray:
        """Validate embedding dimension and dtype."""
        if v.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Embedding must be {EMBEDDING_DIM}-dim, got {v.shape}"
            )
        if v.dtype != np.float32:
            raise ValueError(f"Embedding must be float32, got {v.dtype}")
        return v

    @field_validator("end_line")
    @classmethod
    def validate_line_range(cls, v: int, info) -> int:
        """Ensure end_line >= start_line."""
        if "start_line" in info.data and v < info.data["start_line"]:
            raise ValueError("end_line must be >= start_line")
        return v

    def to_blob(self) -> bytes:
        """Convert embedding to BLOB for SQLite storage."""
        return self.embedding.tobytes()


@dataclass
class Edge:
    """A directed edge in the call graph.

    Attributes:
        caller_id: Symbol ID of the caller
        callee_id: Symbol ID of the callee
        type: Edge type ('call', 'import', 'inherit')
    """

    caller_id: str
    callee_id: str
    type: str  # 'call', 'import', 'inherit'


class Database:
    """SQLite database manager for symbols and edges.

    Handles connection pooling, table creation, and CRUD operations.
    All paths are relative to the repository root.

    Example:
        >>> db = Database(Path(".codegraph/symbols.db"))
        >>> db.initialize()
        >>> symbol = Symbol(..., embedding=np.zeros(768, dtype=np.float32))
        >>> db.insert_symbol(symbol)
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file (created if missing)
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        """Get or create SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def initialize(self) -> None:
        """Create database schema if tables don't exist.

        Creates symbols, edges, and index_meta tables per PRD Section 7.
        """
        conn = self.connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS symbols (
                id TEXT PRIMARY KEY,
                file TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                signature TEXT NOT NULL,
                doc TEXT,
                mutates TEXT,
                embedding BLOB NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edges (
                caller_id TEXT NOT NULL,
                callee_id TEXT NOT NULL,
                type TEXT NOT NULL,
                FOREIGN KEY (caller_id) REFERENCES symbols(id),
                FOREIGN KEY (callee_id) REFERENCES symbols(id)
            );

            CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_symbols_file
                ON symbols(file);

            CREATE INDEX IF NOT EXISTS idx_edges_caller
                ON edges(caller_id);

            CREATE INDEX IF NOT EXISTS idx_edges_callee
                ON edges(callee_id);
        """)
        conn.commit()

    def insert_symbol(self, symbol: Symbol) -> None:
        """Insert a symbol into the database.

        Args:
            symbol: Symbol object with embedding converted to BLOB
        """
        conn = self.connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO symbols
            (id, file, start_line, end_line, signature, doc, mutates, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                symbol.id,
                symbol.file,
                symbol.start_line,
                symbol.end_line,
                symbol.signature,
                symbol.doc,
                symbol.mutates,
                symbol.to_blob(),
            ),
        )
        conn.commit()

    def insert_symbols_batch(self, symbols: List[Symbol]) -> None:
        """Insert multiple symbols in a single transaction.

        Args:
            symbols: List of Symbol objects to insert
        """
        if not symbols:
            return

        conn = self.connect()
        conn.executemany(
            """
            INSERT OR REPLACE INTO symbols
            (id, file, start_line, end_line, signature, doc, mutates, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    s.id,
                    s.file,
                    s.start_line,
                    s.end_line,
                    s.signature,
                    s.doc,
                    s.mutates,
                    s.to_blob(),
                )
                for s in symbols
            ],
        )
        conn.commit()

    def get_symbol(self, symbol_id: str) -> Optional[Symbol]:
        """Retrieve a symbol by ID.

        Args:
            symbol_id: Fully qualified symbol name

        Returns:
            Symbol object if found, None otherwise
        """
        conn = self.connect()
        row = conn.execute(
            "SELECT * FROM symbols WHERE id = ?", (symbol_id,)
        ).fetchone()

        if row is None:
            return None

        return self._row_to_symbol(row)

    def get_all_symbols(self) -> List[Symbol]:
        """Retrieve all symbols from the database.

        Returns:
            List of all Symbol objects
        """
        conn = self.connect()
        rows = conn.execute("SELECT * FROM symbols").fetchall()
        return [self._row_to_symbol(row) for row in rows]

    def delete_symbol(self, symbol_id: str) -> None:
        """Delete a symbol from the database.

        Args:
            symbol_id: Fully qualified symbol name to delete
        """
        conn = self.connect()
        conn.execute("DELETE FROM symbols WHERE id = ?", (symbol_id,))
        conn.commit()

    def delete_symbols_batch(self, symbol_ids: List[str]) -> None:
        """Delete multiple symbols in a single transaction.

        Args:
            symbol_ids: List of symbol IDs to delete
        """
        if not symbol_ids:
            return

        conn = self.connect()
        conn.executemany(
            "DELETE FROM symbols WHERE id = ?", [(sid,) for sid in symbol_ids]
        )
        conn.commit()

    def insert_edge(self, edge: Edge) -> None:
        """Insert an edge into the database.

        Args:
            edge: Edge object representing a call graph relationship
        """
        conn = self.connect()
        conn.execute(
            """
            INSERT INTO edges (caller_id, callee_id, type)
            VALUES (?, ?, ?)
            """,
            (edge.caller_id, edge.callee_id, edge.type),
        )
        conn.commit()

    def insert_edges_batch(self, edges: List[Edge]) -> None:
        """Insert multiple edges in a single transaction.

        Args:
            edges: List of Edge objects to insert
        """
        if not edges:
            return

        conn = self.connect()
        conn.executemany(
            """
            INSERT INTO edges (caller_id, callee_id, type)
            VALUES (?, ?, ?)
            """,
            [(e.caller_id, e.callee_id, e.type) for e in edges],
        )
        conn.commit()

    def get_all_edges(self) -> List[Edge]:
        """Retrieve all edges from the database.

        Returns:
            List of all Edge objects
        """
        conn = self.connect()
        rows = conn.execute("SELECT caller_id, callee_id, type FROM edges").fetchall()
        return [Edge(row["caller_id"], row["callee_id"], row["type"]) for row in rows]

    def set_meta(self, key: str, value: str) -> None:
        """Set a metadata key-value pair.

        Args:
            key: Metadata key (e.g., 'schema_version', 'last_indexed')
            value: String value to store
        """
        conn = self.connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO index_meta (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (key, value),
        )
        conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        """Get a metadata value by key.

        Args:
            key: Metadata key to retrieve

        Returns:
            Value if found, None otherwise
        """
        conn = self.connect()
        row = conn.execute(
            "SELECT value FROM index_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def is_valid(self) -> bool:
        """Check if database is valid and not corrupted.

        Returns:
            True if database is readable, False otherwise
        """
        try:
            conn = self.connect()
            conn.execute("SELECT COUNT(*) FROM symbols")
            conn.execute("SELECT COUNT(*) FROM edges")
            return True
        except sqlite3.DatabaseError:
            return False

    @staticmethod
    def _row_to_symbol(row: sqlite3.Row) -> Symbol:
        """Convert a database row to a Symbol object.

        Args:
            row: SQLite Row object from symbols table

        Returns:
            Symbol instance with embedding deserialized from BLOB
        """
        embedding_blob = row["embedding"]
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)

        # Verify embedding dimension
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"Corrupted embedding: expected {EMBEDDING_DIM} dims, "
                f"got {embedding.shape}"
            )

        return Symbol(
            id=row["id"],
            file=row["file"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            signature=row["signature"],
            doc=row["doc"],
            mutates=row["mutates"],
            embedding=embedding,
        )
