"""
SQLite persistence layer.

Caches all extracted data, classification results, and ontology entries
so the expensive AI pipeline only needs to run once per publication.
On subsequent runs, only new or modified files are processed.
"""

import json
import sqlite3
import logging
from pathlib import Path
from typing import Optional, Any
from dataclasses import asdict

from .classifier import PublicationAnalysis, ClassificationResult, OntologyResult

logger = logging.getLogger(__name__)


class PersistenceStore:
    """SQLite-backed cache for publication analyses."""

    def __init__(self, db_path: str = "library_clerk.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS publications (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_type TEXT NOT NULL,
                title TEXT,
                authors TEXT,  -- JSON array
                year INTEGER,
                abstract TEXT,
                summary TEXT,
                research_domains TEXT,  -- JSON array
                topics TEXT,            -- JSON array
                keywords TEXT,          -- JSON array
                methodologies TEXT,     -- JSON array
                publication_type TEXT,
                key_findings TEXT,      -- JSON array
                raw_stage1 TEXT,        -- Full JSON from stage 1
                raw_stage2 TEXT,        -- Full JSON from stage 2
                ontology_data TEXT,     -- Full ontology result JSON
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS graph_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS cross_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_hash TEXT NOT NULL,
                target_hash TEXT NOT NULL,
                relation TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                properties TEXT,  -- JSON
                FOREIGN KEY (source_hash) REFERENCES publications(file_hash),
                FOREIGN KEY (target_hash) REFERENCES publications(file_hash)
            );

            CREATE INDEX IF NOT EXISTS idx_pub_path ON publications(file_path);
            CREATE INDEX IF NOT EXISTS idx_pub_year ON publications(year);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON cross_edges(source_hash);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON cross_edges(target_hash);
        """)
        self.conn.commit()

    def is_processed(self, file_hash: str) -> bool:
        """Check if a file has already been processed."""
        row = self.conn.execute(
            "SELECT 1 FROM publications WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return row is not None

    def get_processed_hashes(self) -> set[str]:
        """Get set of all already-processed file hashes."""
        rows = self.conn.execute("SELECT file_hash FROM publications").fetchall()
        return {r["file_hash"] for r in rows}

    def store_analysis(self, analysis: PublicationAnalysis) -> None:
        """Store a complete publication analysis."""
        c = analysis.classification
        self.conn.execute(
            """INSERT OR REPLACE INTO publications
            (file_hash, file_path, file_type, title, authors, year,
             abstract, summary, research_domains, topics, keywords,
             methodologies, publication_type, key_findings,
             raw_stage1, raw_stage2, ontology_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                analysis.file_hash,
                analysis.file_path,
                analysis.file_type,
                c.title,
                json.dumps(c.authors),
                c.year,
                c.abstract,
                c.summary,
                json.dumps(c.research_domains),
                json.dumps(c.topics),
                json.dumps(c.keywords),
                json.dumps(c.methodologies),
                c.publication_type,
                json.dumps(c.key_findings),
                json.dumps(analysis.raw_stage1, default=str),
                json.dumps(analysis.raw_stage2, default=str),
                json.dumps(asdict(analysis.ontology), default=str),
            ),
        )
        self.conn.commit()

    def store_cross_edges(self, edges: list[dict]) -> None:
        """Store cross-publication edges."""
        self.conn.execute("DELETE FROM cross_edges")
        for edge in edges:
            self.conn.execute(
                """INSERT INTO cross_edges
                (source_hash, target_hash, relation, weight, properties)
                VALUES (?, ?, ?, ?, ?)""",
                (
                    edge["source"],
                    edge["target"],
                    edge["relation"],
                    edge.get("weight", 1.0),
                    json.dumps(edge.get("properties", {})),
                ),
            )
        self.conn.commit()

    def load_analysis(self, file_hash: str) -> Optional[PublicationAnalysis]:
        """Load a previously stored analysis."""
        row = self.conn.execute(
            "SELECT * FROM publications WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_analysis(row)

    def load_all_analyses(self) -> list[PublicationAnalysis]:
        """Load all stored analyses."""
        rows = self.conn.execute("SELECT * FROM publications ORDER BY year").fetchall()
        return [self._row_to_analysis(r) for r in rows]

    def load_cross_edges(self) -> list[dict]:
        """Load all cross-publication edges."""
        rows = self.conn.execute("SELECT * FROM cross_edges").fetchall()
        return [
            {
                "source": r["source_hash"],
                "target": r["target_hash"],
                "relation": r["relation"],
                "weight": r["weight"],
                "properties": json.loads(r["properties"] or "{}"),
            }
            for r in rows
        ]

    def store_graph_state(self, key: str, value: Any) -> None:
        """Store arbitrary graph state (e.g., full graph JSON)."""
        self.conn.execute(
            """INSERT OR REPLACE INTO graph_state (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)""",
            (key, json.dumps(value, default=str)),
        )
        self.conn.commit()

    def load_graph_state(self, key: str) -> Optional[Any]:
        """Load stored graph state."""
        row = self.conn.execute(
            "SELECT value FROM graph_state WHERE key = ?", (key,)
        ).fetchone()
        if not row:
            return None
        return json.loads(row["value"])

    def get_publications_summary(self) -> list[dict]:
        """Get a summary list of all publications."""
        rows = self.conn.execute(
            """SELECT file_hash, file_path, title, authors, year,
                      publication_type, research_domains, topics, keywords
            FROM publications ORDER BY year"""
        ).fetchall()
        return [
            {
                "file_hash": r["file_hash"],
                "file_path": r["file_path"],
                "title": r["title"],
                "authors": json.loads(r["authors"] or "[]"),
                "year": r["year"],
                "publication_type": r["publication_type"],
                "research_domains": json.loads(r["research_domains"] or "[]"),
                "topics": json.loads(r["topics"] or "[]"),
                "keywords": json.loads(r["keywords"] or "[]"),
            }
            for r in rows
        ]

    def search(self, query: str) -> list[dict]:
        """Full-text search across titles, abstracts, keywords."""
        q = f"%{query}%"
        rows = self.conn.execute(
            """SELECT file_hash, title, authors, year, abstract, keywords
            FROM publications
            WHERE title LIKE ? OR abstract LIKE ? OR keywords LIKE ?
            ORDER BY year""",
            (q, q, q),
        ).fetchall()
        return [
            {
                "file_hash": r["file_hash"],
                "title": r["title"],
                "authors": json.loads(r["authors"] or "[]"),
                "year": r["year"],
                "abstract": r["abstract"],
                "keywords": json.loads(r["keywords"] or "[]"),
            }
            for r in rows
        ]

    def _row_to_analysis(self, row) -> PublicationAnalysis:
        classification = ClassificationResult(
            title=row["title"] or "",
            authors=json.loads(row["authors"] or "[]"),
            year=row["year"],
            abstract=row["abstract"] or "",
            summary=row["summary"] or "",
            research_domains=json.loads(row["research_domains"] or "[]"),
            topics=json.loads(row["topics"] or "[]"),
            keywords=json.loads(row["keywords"] or "[]"),
            methodologies=json.loads(row["methodologies"] or "[]"),
            publication_type=row["publication_type"] or "",
            key_findings=json.loads(row["key_findings"] or "[]"),
        )

        ontology_data = json.loads(row["ontology_data"] or "{}")
        ontology = OntologyResult(
            concepts=ontology_data.get("concepts", []),
            concept_relationships=ontology_data.get("concept_relationships", []),
            topic_hierarchy=ontology_data.get("topic_hierarchy", []),
            suggested_connections=ontology_data.get("suggested_connections", []),
            graph_nodes=ontology_data.get("graph_nodes", []),
            graph_edges=ontology_data.get("graph_edges", []),
        )

        return PublicationAnalysis(
            file_path=row["file_path"],
            file_hash=row["file_hash"],
            file_type=row["file_type"],
            classification=classification,
            ontology=ontology,
            raw_stage1=json.loads(row["raw_stage1"] or "{}"),
            raw_stage2=json.loads(row["raw_stage2"] or "{}"),
        )

    def get_stats(self) -> dict:
        pub_count = self.conn.execute(
            "SELECT count(*) as c FROM publications"
        ).fetchone()["c"]
        edge_count = self.conn.execute(
            "SELECT count(*) as c FROM cross_edges"
        ).fetchone()["c"]

        year_dist = {}
        for r in self.conn.execute(
            "SELECT year, count(*) as c FROM publications WHERE year IS NOT NULL GROUP BY year ORDER BY year"
        ).fetchall():
            year_dist[r["year"]] = r["c"]

        type_dist = {}
        for r in self.conn.execute(
            "SELECT publication_type, count(*) as c FROM publications GROUP BY publication_type"
        ).fetchall():
            type_dist[r["publication_type"] or "unknown"] = r["c"]

        return {
            "publications": pub_count,
            "cross_edges": edge_count,
            "year_distribution": year_dist,
            "type_distribution": type_dist,
        }

    def close(self):
        self.conn.close()
