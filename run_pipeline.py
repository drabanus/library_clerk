#!/usr/bin/env python3
"""
Library Clerk - Publication Classification Pipeline

Scans a library of PDFs and ebooks, classifies them using local LLMs via Ollama,
builds a knowledge graph with meaningful ontologies, and persists everything to
a SQLite database for later visualization and querying.

Usage:
    python run_pipeline.py                      # Use default config.yaml
    python run_pipeline.py --config my.yaml     # Custom config
    python run_pipeline.py --paths ~/papers     # Override library paths
    python run_pipeline.py --limit 10           # Process only the first 10 files
    python run_pipeline.py --solo-file 'My Paper.pdf'  # Process a single file
    python run_pipeline.py --reprocess          # Force reprocess all files
    python run_pipeline.py --dry-run            # Scan only, don't classify

The pipeline runs two LLM stages sequentially per document:
  Stage 1 (ministral-3:3b):  Classification, keywords, metadata extraction
  Stage 2 (deepseek-coder:6.7b): Ontology entries, graph relationships

After individual classification, cross-publication edges are computed
(shared keywords, topics, co-authorship, temporal succession, etc.)
and the full graph is persisted.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

from src.extractor import scan_library, extract_document, ExtractedDocument
from src.ollama_client import OllamaClient
from src.classifier import PublicationClassifier, PublicationAnalysis
from src.graph_manager import create_backend, GraphBackend
from src.persistence import PersistenceStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("library_clerk")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_graph_from_analyses(
    graph: GraphBackend,
    analyses: list[PublicationAnalysis],
    cross_edges: list[dict],
) -> None:
    """Populate graph database from classification results."""
    logger.info("Building knowledge graph...")

    for analysis in analyses:
        c = analysis.classification
        o = analysis.ontology
        pub_id = f"pub_{analysis.file_hash[:12]}"

        # Publication node
        graph.add_node(pub_id, "Publication", {
            "label": c.title,
            "title": c.title,
            "year": c.year,
            "publication_type": c.publication_type,
            "summary": c.summary,
            "abstract": c.abstract,
            "file_path": analysis.file_path,
            "file_hash": analysis.file_hash,
        })

        # Author nodes + edges
        for author in c.authors:
            author_id = f"author_{author.lower().replace(' ', '_')}"
            graph.add_node(author_id, "Author", {"label": author, "name": author})
            graph.add_edge(pub_id, author_id, "AUTHORED_BY", weight=1.0)

        # Domain nodes + edges
        for domain in c.research_domains:
            domain_id = f"domain_{domain.lower().replace(' ', '_')}"
            graph.add_node(domain_id, "Domain", {"label": domain, "name": domain})
            graph.add_edge(pub_id, domain_id, "IN_DOMAIN", weight=0.9)

        # Topic nodes + edges
        for topic in c.topics:
            topic_id = f"topic_{topic.lower().replace(' ', '_')}"
            graph.add_node(topic_id, "Topic", {"label": topic, "name": topic})
            graph.add_edge(pub_id, topic_id, "BELONGS_TO", weight=0.8)

        # Keyword nodes + edges
        for kw in c.keywords:
            kw_id = f"kw_{kw.lower().replace(' ', '_')}"
            graph.add_node(kw_id, "Keyword", {"label": kw, "term": kw})
            graph.add_edge(pub_id, kw_id, "HAS_KEYWORD", weight=0.6)

        # Methodology nodes + edges
        for method in c.methodologies:
            method_id = f"method_{method.lower().replace(' ', '_')}"
            graph.add_node(method_id, "Methodology", {
                "label": method, "name": method,
            })
            graph.add_edge(pub_id, method_id, "USES_METHODOLOGY", weight=0.7)

        # Ontology-generated concepts
        for concept in o.concepts:
            cname = concept.get("name", "")
            if not cname:
                continue
            concept_id = f"concept_{cname.lower().replace(' ', '_')}"
            graph.add_node(concept_id, "Concept", {
                "label": cname,
                "definition": concept.get("definition", ""),
                "domain": concept.get("domain", ""),
            })
            graph.add_edge(pub_id, concept_id, "DISCUSSES", weight=0.7)

        # Concept relationships from ontology
        for rel in o.concept_relationships:
            from_id = f"concept_{rel.get('from', '').lower().replace(' ', '_')}"
            to_id = f"concept_{rel.get('to', '').lower().replace(' ', '_')}"
            relation = rel.get("relation", "RELATED_CONCEPT")
            if from_id and to_id:
                graph.add_edge(from_id, to_id, relation.upper(), weight=0.6)

        # Topic hierarchy from ontology
        for th in o.topic_hierarchy:
            topic = th.get("topic", "")
            parent = th.get("parent", "")
            if topic and parent:
                child_id = f"topic_{topic.lower().replace(' ', '_')}"
                parent_id = f"topic_{parent.lower().replace(' ', '_')}"
                graph.add_node(child_id, "Topic", {"label": topic, "name": topic})
                graph.add_node(parent_id, "Topic", {"label": parent, "name": parent})
                graph.add_edge(child_id, parent_id, "SUBTOPIC_OF", weight=0.8)

        # Additional graph nodes/edges from ontology model
        for gn in o.graph_nodes:
            nid = gn.get("id", "")
            if nid and not graph.get_node(nid):
                graph.add_node(nid, gn.get("type", "Entity"), {
                    "label": gn.get("label", nid),
                    **gn.get("properties", {}),
                })

        for ge in o.graph_edges:
            src = ge.get("source", "")
            tgt = ge.get("target", "")
            if src and tgt:
                graph.add_edge(
                    src, tgt,
                    ge.get("relation", "RELATED"),
                    weight=ge.get("weight", 0.5),
                    properties=ge.get("properties"),
                )

    # Cross-publication edges
    hash_to_pubid = {a.file_hash: f"pub_{a.file_hash[:12]}" for a in analyses}
    for edge in cross_edges:
        src = hash_to_pubid.get(edge["source"])
        tgt = hash_to_pubid.get(edge["target"])
        if src and tgt:
            graph.add_edge(
                src, tgt,
                edge["relation"],
                weight=edge.get("weight", 0.5),
                properties=edge.get("properties"),
            )

    stats = graph.get_stats()
    logger.info(
        f"Graph built: {stats['total_nodes']} nodes, {stats['total_edges']} edges"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Library Clerk - Publication Classification Pipeline"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--paths", nargs="+", help="Override library paths from config"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Limit processing to the first N files",
    )
    parser.add_argument(
        "--solo-file",
        help="Process a single file by name (filename or full path)",
    )
    parser.add_argument(
        "--reprocess", action="store_true",
        help="Force reprocess all files, ignoring cache",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan library and show what would be processed",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    # Resolve library paths
    lib_paths = args.paths or config.get("library", {}).get("paths", [])
    extensions = config.get("library", {}).get("extensions", [".pdf", ".epub"])

    # --solo-file: process a single file by name or path
    if args.solo_file:
        solo = Path(args.solo_file)
        if solo.is_file():
            # Argument is a direct path
            files = [str(solo.resolve())]
        else:
            # Search for the filename inside the library paths
            logger.info(f"Searching library for '{args.solo_file}'...")
            all_files = scan_library(lib_paths, extensions)
            target = args.solo_file
            files = [
                f for f in all_files
                if Path(f).name == target or target in f
            ]
            if not files:
                logger.error(
                    f"File '{args.solo_file}' not found in library paths: {lib_paths}"
                )
                sys.exit(1)
            if len(files) > 1:
                logger.warning(
                    f"Multiple matches for '{args.solo_file}', using first:"
                )
                for f in files:
                    logger.warning(f"  {f}")
                files = files[:1]
        logger.info(f"Solo file: {files[0]}")
    else:
        logger.info(f"Scanning library paths: {lib_paths}")
        files = scan_library(lib_paths, extensions)
        logger.info(f"Found {len(files)} publication files")

    if not files:
        logger.warning("No files found. Check your library paths in config.yaml")
        sys.exit(0)

    # --limit: cap the number of files to process
    if args.limit > 0 and not args.solo_file:
        logger.info(f"Limiting to first {args.limit} of {len(files)} files")
        files = files[:args.limit]

    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN - {len(files)} file(s) to process:")
        print(f"{'='*60}")
        for f in files:
            print(f"  {f}")
        sys.exit(0)

    # Initialize components
    ollama_cfg = config.get("ollama", {})
    client = OllamaClient(
        base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
        timeout=ollama_cfg.get("timeout", 120),
    )

    # Report compute hardware
    logger.info(f"Compute: {client.gpu_info.summary}")
    if not client.gpu_info.has_gpu:
        logger.warning("No GPU found — using CPU inference (this will be slower)")

    # Check model availability
    classifier_model = ollama_cfg.get("classifier_model", "ministral-3:3b")
    ontology_model = ollama_cfg.get("ontology_model", "deepseek-coder:6.7b")

    availability = client.check_models([classifier_model, ontology_model])
    for model, available in availability.items():
        status = "available" if available else "NOT FOUND"
        logger.info(f"Model {model}: {status}")
        if not available:
            logger.error(
                f"Model {model} not available. Pull it with: ollama pull {model}"
            )
            sys.exit(1)

    classifier = PublicationClassifier(
        client=client,
        classifier_model=classifier_model,
        ontology_model=ontology_model,
    )

    persistence_cfg = config.get("persistence", {})
    store = PersistenceStore(
        db_path=persistence_cfg.get("database", "library_clerk.db")
    )

    graph = create_backend(config.get("graph", {}))

    # Determine which files need processing
    processed_hashes = set() if args.reprocess else store.get_processed_hashes()

    # Extract and classify
    analyses: list[PublicationAnalysis] = []
    new_count = 0
    cached_count = 0
    ocr_count = 0
    error_count = 0

    start_time = time.time()

    for file_path in tqdm(files, desc="Processing publications"):
        try:
            doc = extract_document(file_path)

            if doc.was_ocred:
                ocr_count += 1

            if doc.file_hash in processed_hashes:
                # Load from cache
                cached = store.load_analysis(doc.file_hash)
                if cached:
                    analyses.append(cached)
                    cached_count += 1
                    continue

            if doc.word_count < 50:
                logger.warning(
                    f"Skipping {file_path}: too little text extracted"
                    f"{' (even after OCR)' if doc.was_ocred else ''}"
                )
                continue

            # Run classification pipeline
            analysis = classifier.classify(doc)
            analyses.append(analysis)
            store.store_analysis(analysis)
            new_count += 1

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            error_count += 1

    elapsed = time.time() - start_time
    logger.info(
        f"Classification complete: {new_count} new, {cached_count} cached, "
        f"{ocr_count} OCR'd, {error_count} errors ({elapsed:.1f}s)"
    )

    # Compute cross-publication edges
    if analyses:
        logger.info("Computing cross-publication relationships...")
        cross_edges = classifier.compute_cross_publication_edges(analyses)
        store.store_cross_edges(cross_edges)
        logger.info(f"Found {len(cross_edges)} cross-publication edges")

        # Build graph
        build_graph_from_analyses(graph, analyses, cross_edges)

        # Persist graph state
        graph_data = graph.export_json()
        store.store_graph_state("full_graph", graph_data)

    # Print summary
    db_stats = store.get_stats()
    print(f"\n{'='*60}")
    print("LIBRARY CLERK - PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Publications processed: {db_stats['publications']}")
    print(f"  Scanned PDFs OCR'd: {ocr_count}")
    print(f"  Cross-publication edges: {db_stats['cross_edges']}")
    if db_stats['year_distribution']:
        years = sorted(db_stats['year_distribution'].keys())
        print(f"  Year range: {years[0]} - {years[-1]}")
    if db_stats['type_distribution']:
        print(f"  Types: {db_stats['type_distribution']}")
    graph_stats = graph.get_stats()
    print(f"  Graph nodes: {graph_stats['total_nodes']}")
    print(f"  Graph edges: {graph_stats['total_edges']}")
    print(f"  Node types: {graph_stats['node_types']}")
    print(f"  LLM stats: {client.stats}")
    print(f"\nRun the visualization server:")
    print(f"  python run_server.py")
    print(f"{'='*60}\n")

    store.close()


if __name__ == "__main__":
    main()
