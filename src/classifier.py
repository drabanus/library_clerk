"""
Sequential two-stage LLM classification pipeline.

Stage 1 (ministral-3:3b):
  - Extract/refine title, authors, year
  - Classify into research domains and topics
  - Generate keywords
  - Identify methodologies
  - Produce a concise summary

Stage 2 (deepseek-coder:6.7b):
  - Generate structured ontology entries (nodes + edges)
  - Define concepts and their hierarchical relationships
  - Map cross-publication relationships
  - Produce graph-ready JSON
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .ollama_client import OllamaClient  # noqa: F401 — backward compat
from .llm_backend import LLMBackend
from .extractor import ExtractedDocument

logger = logging.getLogger(__name__)

# Maximum characters to send to the LLM (roughly ~4K tokens)
MAX_TEXT_CHARS = 12000


@dataclass
class ClassificationResult:
    """Output of Stage 1: ministral-3:3b classification."""
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    summary: str = ""
    research_domains: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    methodologies: list[str] = field(default_factory=list)
    publication_type: str = ""  # paper, book, thesis, report, etc.
    key_findings: list[str] = field(default_factory=list)


@dataclass
class OntologyResult:
    """Output of Stage 2: deepseek-coder:6.7b ontology generation."""
    concepts: list[dict[str, str]] = field(default_factory=list)
    concept_relationships: list[dict[str, str]] = field(default_factory=list)
    topic_hierarchy: list[dict[str, str]] = field(default_factory=list)
    suggested_connections: list[dict[str, Any]] = field(default_factory=list)
    graph_nodes: list[dict[str, Any]] = field(default_factory=list)
    graph_edges: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PublicationAnalysis:
    """Complete analysis result for a single publication."""
    file_path: str
    file_hash: str
    file_type: str
    classification: ClassificationResult
    ontology: OntologyResult
    raw_stage1: dict = field(default_factory=dict)
    raw_stage2: dict = field(default_factory=dict)


def _truncate_text(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    """Truncate text intelligently, keeping beginning and end."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[... content truncated ...]\n\n" + text[-half:]


STAGE1_SYSTEM = """You are a research librarian AI. You analyze academic publications and extract structured metadata.
Always respond with valid JSON. Be precise and factual. Extract information directly from the text provided."""

STAGE1_PROMPT = """Analyze this publication text and extract the following metadata as JSON:

{{
  "title": "extracted or inferred title",
  "authors": ["author1", "author2"],
  "year": 2024,
  "abstract": "the abstract or a generated one if missing (2-3 sentences)",
  "summary": "a concise summary of the main content (3-5 sentences)",
  "research_domains": ["broad field 1", "broad field 2"],
  "topics": ["specific topic 1", "specific topic 2", "specific topic 3"],
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "methodologies": ["methodology1", "methodology2"],
  "publication_type": "paper|book|thesis|report|review|tutorial",
  "key_findings": ["finding1", "finding2"]
}}

Rules:
- research_domains: 1-3 broad fields (e.g., "Machine Learning", "Quantum Physics", "Neuroscience")
- topics: 2-5 specific topics within those domains
- keywords: 5-10 descriptive keywords
- methodologies: analytical methods, experimental approaches, or theoretical frameworks used
- publication_type: classify as one of paper, book, thesis, report, review, tutorial, or other
- If metadata is already in the document (title, authors), prefer the document's own metadata
- If year cannot be determined, use null

PUBLICATION TEXT:
{text}"""


STAGE2_SYSTEM = """You are a strict ontology extraction engine. You extract structured semantic triples from scientific publication text following a fixed schema. Always respond with valid JSON only. Be conservative: extract only observable facts stated in the text. If uncertain, extract less, not more."""

STAGE2_MAX_TEXT_CHARS = 8000

STAGE2_PROMPT = """Extract a structured ontology from this publication.

METADATA (from prior analysis):
- Title: {title}
- Authors: {authors}
- Year: {year}
- Domains: {domains}
- Summary: {summary}

DOCUMENT TEXT (may be truncated):
{text}

────────────────────────────────────────

ONTOLOGY EXTRACTION RULES

1. Extract only observable facts. Do NOT infer intent, correctness, novelty, or importance.
2. Distinguish "says" from "is" — use makes_claim, reports_result, defines, etc.
   NEVER use predicates like "proves", "is true", "is wrong".
3. One document at a time — assume no global knowledge beyond what is written.
4. Prefer atomic triples — decompose complex statements into simple triples.
5. Be conservative — if uncertain, extract less. Aim for 5-15 nodes and 10-30 triples.

CANONICAL NODE TYPES (use only these):
  Publication, Person, Organization, Section, Figure, Table, Equation,
  Reference, Concept, Method, Instrument, Dataset, Result, Claim

NODE ID FORMAT: type:lowercase_slug
  Examples: concept:neural_networks, method:gradient_descent, person:j_doe,
  sec:introduction, ref:smith_2020, result:accuracy_95pct, claim:01
Use "pub:this" to refer to the current publication.

ALLOWED PREDICATES (use only these):

Identity & provenance:
  has_title, has_abstract, has_identifier, has_doi, has_publication_date

Authorship:
  has_author, has_affiliation, affiliated_with, funded_by, acknowledges

Document structure:
  has_section, has_subsection, has_figure, has_table, has_equation, has_page_range

Citations:
  cites, self_cites

Concepts & terminology:
  defines, uses_term, aliases, abbreviates, refers_to_concept

Methods & instrumentation:
  uses_method, describes_method, uses_instrument, uses_software,
  uses_algorithm, has_parameter, has_value

Results & data:
  reports_result, reports_measurement, reports_dataset,
  has_numeric_value, has_unit, has_uncertainty, derived_using_method

Claims & reasoning:
  makes_claim, supports_claim, qualifies_claim, limits_claim,
  contradicts_claim, assumes, concludes

Comparison & relation:
  compares_with, extends, improves_upon, reproduces,
  is_consistent_with, is_inconsistent_with

Scope:
  applies_to, limited_to, states_limitation, states_uncertainty, states_future_work

EXTRACTION PROCEDURE (follow in order):
1. Create Section nodes for major document sections.
2. Create Concept nodes for explicitly defined or named concepts.
3. Create Method/Instrument nodes for named methods, algorithms, tools.
4. Create Result nodes for reported results (with numeric values when available).
5. Create Claim nodes for key claims and conclusions.
6. Create Reference nodes for cited works.
7. Connect all nodes with triples using only allowed predicates.

RESPOND WITH JSON ONLY:
{{
  "nodes": [
    {{"id": "concept:example", "type": "Concept", "label": "Display Name", "properties": {{}}}},
    {{"id": "method:example", "type": "Method", "label": "Example Method"}}
  ],
  "triples": [
    {{"subject": "pub:this", "predicate": "defines", "object": "concept:example"}},
    {{"subject": "pub:this", "predicate": "uses_method", "object": "method:example"}}
  ]
}}"""


class SkipFile(Exception):
    """Raised when the user requests skipping the current file."""


class PublicationClassifier:
    """Two-stage sequential classification pipeline."""

    def __init__(
        self,
        client: LLMBackend,
        classifier_model: str = "ministral-3:3b",
        ontology_model: str = "deepseek-coder:6.7b",
    ):
        self.client = client
        self.classifier_model = classifier_model
        self.ontology_model = ontology_model

    def classify(
        self,
        doc: ExtractedDocument,
        on_stage: Any | None = None,
        check_skip: Any | None = None,
    ) -> PublicationAnalysis:
        """
        Run full two-stage classification pipeline on a document.

        Stage 1: Classification with classifier_model
        Stage 2: Ontology generation with ontology_model

        Args:
            on_stage: optional callback(stage_name: str) for progress reporting.
            check_skip: optional callable() -> bool; if it returns True between
                        stages the file is skipped by raising SkipFile.
        """
        logger.info(f"Classifying: {doc.file_path}")

        # Stage 1: Classification
        if on_stage:
            on_stage("Stage 1: classifying")
        t0 = time.monotonic()
        classification, raw_s1 = self._stage1_classify(doc)
        s1_secs = time.monotonic() - t0
        logger.info(
            f"  Stage 1 complete ({s1_secs:.1f}s): {classification.title} "
            f"({len(classification.keywords)} keywords, "
            f"{len(classification.topics)} topics)"
        )

        # Check for skip request between stages
        if check_skip and check_skip():
            raise SkipFile(doc.file_path)

        # Stage 2: Ontology generation
        if on_stage:
            on_stage("Stage 2: ontology")
        t1 = time.monotonic()
        ontology, raw_s2 = self._stage2_ontology(classification, doc.raw_text)
        s2_secs = time.monotonic() - t1
        logger.info(
            f"  Stage 2 complete ({s2_secs:.1f}s): {len(ontology.graph_nodes)} nodes, "
            f"{len(ontology.graph_edges)} edges"
        )

        return PublicationAnalysis(
            file_path=doc.file_path,
            file_hash=doc.file_hash,
            file_type=doc.file_type,
            classification=classification,
            ontology=ontology,
            raw_stage1=raw_s1,
            raw_stage2=raw_s2,
        )

    def _stage1_classify(
        self, doc: ExtractedDocument
    ) -> tuple[ClassificationResult, dict]:
        """Stage 1: Use ministral-3:3b for classification and keyword extraction."""
        text_chunk = _truncate_text(doc.raw_text)
        prompt = STAGE1_PROMPT.format(text=text_chunk)

        data = self.client.generate_json(
            model=self.classifier_model,
            prompt=prompt,
            system=STAGE1_SYSTEM,
            temperature=0.2,
        )

        result = ClassificationResult(
            title=data.get("title") or doc.title or "Unknown",
            authors=data.get("authors") or doc.authors or [],
            year=data.get("year") or doc.year,
            abstract=data.get("abstract", ""),
            summary=data.get("summary", ""),
            research_domains=data.get("research_domains", []),
            topics=data.get("topics", []),
            keywords=data.get("keywords", []),
            methodologies=data.get("methodologies", []),
            publication_type=data.get("publication_type", "paper"),
            key_findings=data.get("key_findings", []),
        )

        # Merge document metadata if LLM missed it
        if not result.authors and doc.authors:
            result.authors = doc.authors
        if not result.year and doc.year:
            result.year = doc.year
        if (not result.title or result.title == "Unknown") and doc.title:
            result.title = doc.title

        return result, data

    def _stage2_ontology(
        self, classification: ClassificationResult, doc_text: str = ""
    ) -> tuple[OntologyResult, dict]:
        """Stage 2: Strict ontology extraction with canonical predicates."""
        text_chunk = _truncate_text(doc_text, STAGE2_MAX_TEXT_CHARS) if doc_text else ""

        prompt = STAGE2_PROMPT.format(
            title=classification.title,
            authors=", ".join(classification.authors),
            year=classification.year or "unknown",
            domains=", ".join(classification.research_domains),
            summary=classification.summary,
            text=text_chunk,
        )

        data = self.client.generate_json(
            model=self.ontology_model,
            prompt=prompt,
            system=STAGE2_SYSTEM,
            temperature=0.2,
        )

        # New format: "nodes" + "triples"
        raw_nodes = data.get("nodes", [])
        raw_triples = data.get("triples", [])

        # Normalize triples → graph_edges (source/target/relation)
        graph_edges = []
        for t in raw_triples:
            src = t.get("subject", "")
            tgt = t.get("object", "")
            pred = t.get("predicate", "RELATED")
            if src and tgt and pred:
                graph_edges.append({
                    "source": src,
                    "target": tgt,
                    "relation": pred,
                    "weight": t.get("weight", 0.7),
                    "properties": t.get("properties", {}),
                })

        # Fallback: support legacy format from cached results
        graph_nodes = raw_nodes or data.get("graph_nodes", [])
        if not graph_edges:
            graph_edges = data.get("graph_edges", [])

        result = OntologyResult(
            concepts=data.get("concepts", []),
            concept_relationships=data.get("concept_relationships", []),
            topic_hierarchy=data.get("topic_hierarchy", []),
            suggested_connections=data.get("suggested_connections", []),
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
        )

        return result, data

    def compute_cross_publication_edges(
        self, analyses: list[PublicationAnalysis]
    ) -> list[dict]:
        """
        Compute edges between publications based on shared entities.
        This runs after all individual publications are classified.
        """
        edges = []

        for i, a in enumerate(analyses):
            kw_i = set(k.lower() for k in a.classification.keywords)
            topics_i = set(t.lower() for t in a.classification.topics)
            domains_i = set(d.lower() for d in a.classification.research_domains)
            methods_i = set(m.lower() for m in a.classification.methodologies)
            authors_i = set(au.lower() for au in a.classification.authors)

            for j, b in enumerate(analyses):
                if j <= i:
                    continue

                kw_j = set(k.lower() for k in b.classification.keywords)
                topics_j = set(t.lower() for t in b.classification.topics)
                domains_j = set(d.lower() for d in b.classification.research_domains)
                methods_j = set(m.lower() for m in b.classification.methodologies)
                authors_j = set(au.lower() for au in b.classification.authors)

                # Keyword overlap
                kw_shared = kw_i & kw_j
                if kw_shared:
                    weight = len(kw_shared) / max(
                        len(kw_i | kw_j), 1
                    )
                    edges.append({
                        "source": a.file_hash,
                        "target": b.file_hash,
                        "relation": "SHARES_KEYWORDS",
                        "weight": round(weight, 3),
                        "properties": {
                            "shared_keywords": sorted(kw_shared)
                        },
                    })

                # Topic overlap
                topic_shared = topics_i & topics_j
                if topic_shared:
                    weight = len(topic_shared) / max(
                        len(topics_i | topics_j), 1
                    )
                    edges.append({
                        "source": a.file_hash,
                        "target": b.file_hash,
                        "relation": "SHARES_TOPICS",
                        "weight": round(weight, 3),
                        "properties": {
                            "shared_topics": sorted(topic_shared)
                        },
                    })

                # Domain overlap
                domain_shared = domains_i & domains_j
                if domain_shared:
                    weight = len(domain_shared) / max(
                        len(domains_i | domains_j), 1
                    )
                    edges.append({
                        "source": a.file_hash,
                        "target": b.file_hash,
                        "relation": "SHARES_DOMAIN",
                        "weight": round(weight, 3),
                        "properties": {
                            "shared_domains": sorted(domain_shared)
                        },
                    })

                # Methodology overlap
                method_shared = methods_i & methods_j
                if method_shared:
                    edges.append({
                        "source": a.file_hash,
                        "target": b.file_hash,
                        "relation": "SHARES_METHODOLOGY",
                        "weight": round(
                            len(method_shared) / max(len(methods_i | methods_j), 1), 3
                        ),
                        "properties": {
                            "shared_methods": sorted(method_shared)
                        },
                    })

                # Co-authorship
                author_shared = authors_i & authors_j
                if author_shared:
                    edges.append({
                        "source": a.file_hash,
                        "target": b.file_hash,
                        "relation": "CO_AUTHORED",
                        "weight": 1.0,
                        "properties": {
                            "shared_authors": sorted(author_shared)
                        },
                    })

                # Temporal succession (same domain, chronological)
                if domain_shared and a.classification.year and b.classification.year:
                    if a.classification.year < b.classification.year:
                        edges.append({
                            "source": a.file_hash,
                            "target": b.file_hash,
                            "relation": "TEMPORAL_PREDECESSOR",
                            "weight": 0.5,
                            "properties": {
                                "year_from": a.classification.year,
                                "year_to": b.classification.year,
                            },
                        })

        return edges
