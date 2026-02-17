"""
Query engine for the knowledge base.

Provides semantic search, path finding, clustering analysis,
research timeline, and recommendation capabilities.
"""

import logging
from collections import Counter, defaultdict
from typing import Any, Optional

from .graph_manager import GraphBackend
from .persistence import PersistenceStore

logger = logging.getLogger(__name__)


class QueryEngine:
    """High-level query interface over the knowledge graph and persistence store."""

    def __init__(self, graph: GraphBackend, store: PersistenceStore):
        self.graph = graph
        self.store = store

    def search(self, query: str, node_type: str | None = None) -> dict:
        """
        Search across the knowledge base.
        Returns matching graph nodes and publications.
        """
        graph_results = self.graph.search_nodes(query, node_type)
        pub_results = self.store.search(query)

        return {
            "graph_nodes": graph_results[:20],
            "publications": pub_results[:20],
            "query": query,
        }

    def get_publication_detail(self, file_hash: str) -> dict | None:
        """Get full details for a publication including its graph neighborhood."""
        analysis = self.store.load_analysis(file_hash)
        if not analysis:
            return None

        pub_id = f"pub_{file_hash[:12]}"
        neighborhood = self.graph.get_subgraph(pub_id, depth=1)

        return {
            "publication": {
                "title": analysis.classification.title,
                "authors": analysis.classification.authors,
                "year": analysis.classification.year,
                "abstract": analysis.classification.abstract,
                "summary": analysis.classification.summary,
                "domains": analysis.classification.research_domains,
                "topics": analysis.classification.topics,
                "keywords": analysis.classification.keywords,
                "methodologies": analysis.classification.methodologies,
                "type": analysis.classification.publication_type,
                "findings": analysis.classification.key_findings,
                "file_path": analysis.file_path,
            },
            "neighborhood": neighborhood,
        }

    def find_related(self, file_hash: str, limit: int = 10) -> list[dict]:
        """Find publications most related to a given one."""
        cross_edges = self.store.load_cross_edges()

        # Aggregate edge weights between this publication and all others
        scores: dict[str, float] = defaultdict(float)
        relations: dict[str, list[str]] = defaultdict(list)

        for edge in cross_edges:
            other = None
            if edge["source"] == file_hash:
                other = edge["target"]
            elif edge["target"] == file_hash:
                other = edge["source"]
            if other:
                scores[other] += edge.get("weight", 0.5)
                relations[other].append(edge["relation"])

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:limit]

        results = []
        for other_hash, score in ranked:
            analysis = self.store.load_analysis(other_hash)
            if analysis:
                results.append({
                    "file_hash": other_hash,
                    "title": analysis.classification.title,
                    "year": analysis.classification.year,
                    "score": round(score, 3),
                    "relation_types": list(set(relations[other_hash])),
                })

        return results

    def get_publications_for_node(self, node_id: str) -> list[dict]:
        """
        Find all Publication nodes connected to a given node.

        Walks the node's neighbors looking for Publication nodes,
        then enriches each with title/authors/year from the store.
        """
        neighbors = self.graph.get_neighbors(node_id)
        pub_neighbors = [
            n for n in neighbors
            if n.get("node_type") == "Publication"
        ]

        results = []
        seen = set()
        for nb in pub_neighbors:
            pub_id = nb["id"]
            if pub_id in seen:
                continue
            seen.add(pub_id)
            # Extract file_hash from the pub node id (pub_<hash12>)
            file_hash = nb.get("file_hash")
            if not file_hash and pub_id.startswith("pub_"):
                hash_prefix = pub_id[4:]  # after "pub_"
                # Look up in the store by prefix
                for summary in self.store.get_publications_summary():
                    if summary["file_hash"].startswith(hash_prefix):
                        file_hash = summary["file_hash"]
                        break

            title = nb.get("title") or nb.get("label") or pub_id
            entry = {
                "node_id": pub_id,
                "title": title,
                "relation": nb.get("_edge_relation", ""),
            }

            # Enrich from persistence if possible
            if file_hash:
                analysis = self.store.load_analysis(file_hash)
                if analysis:
                    entry["file_hash"] = file_hash
                    entry["title"] = analysis.classification.title
                    entry["authors"] = analysis.classification.authors
                    entry["year"] = analysis.classification.year

            results.append(entry)

        return results

    def get_clusters(self) -> list[dict]:
        """
        Identify clusters of related publications based on shared topics/domains.
        """
        analyses = self.store.load_all_analyses()

        # Group by domain
        domain_groups: dict[str, list[dict]] = defaultdict(list)
        for a in analyses:
            for domain in a.classification.research_domains:
                domain_groups[domain.lower()].append({
                    "file_hash": a.file_hash,
                    "title": a.classification.title,
                    "year": a.classification.year,
                    "topics": a.classification.topics,
                })

        clusters = []
        for domain, pubs in sorted(domain_groups.items(), key=lambda x: -len(x[1])):
            clusters.append({
                "domain": domain,
                "size": len(pubs),
                "publications": pubs,
                "year_range": (
                    min((p["year"] for p in pubs if p["year"]), default=None),
                    max((p["year"] for p in pubs if p["year"]), default=None),
                ),
                "common_topics": _most_common(
                    [t for p in pubs for t in p["topics"]], n=5
                ),
            })

        return clusters

    def get_research_timeline(self) -> list[dict]:
        """
        Build a chronological timeline of research activity.
        Groups publications by year with domain evolution.
        """
        analyses = self.store.load_all_analyses()

        by_year: dict[int, list] = defaultdict(list)
        for a in analyses:
            if a.classification.year:
                by_year[a.classification.year].append(a)

        timeline = []
        for year in sorted(by_year.keys()):
            pubs = by_year[year]
            domains = Counter()
            topics = Counter()
            for a in pubs:
                for d in a.classification.research_domains:
                    domains[d] += 1
                for t in a.classification.topics:
                    topics[t] += 1

            timeline.append({
                "year": year,
                "count": len(pubs),
                "publications": [
                    {"title": a.classification.title, "file_hash": a.file_hash}
                    for a in pubs
                ],
                "domains": dict(domains.most_common(5)),
                "topics": dict(topics.most_common(5)),
            })

        return timeline

    def get_keyword_network(self) -> dict:
        """
        Build a keyword co-occurrence network.
        Keywords that appear together in publications are connected.
        """
        analyses = self.store.load_all_analyses()

        # Count co-occurrences
        cooccurrence: dict[tuple, int] = Counter()
        keyword_count: dict[str, int] = Counter()

        for a in analyses:
            kws = [k.lower() for k in a.classification.keywords]
            for kw in kws:
                keyword_count[kw] += 1
            for i, kw1 in enumerate(kws):
                for kw2 in kws[i + 1:]:
                    pair = tuple(sorted([kw1, kw2]))
                    cooccurrence[pair] += 1

        # Build vis-ready network
        nodes = [
            {"id": kw, "label": kw, "size": count}
            for kw, count in keyword_count.most_common(100)
        ]
        node_ids = {n["id"] for n in nodes}

        edges = [
            {"source": pair[0], "target": pair[1], "weight": count}
            for pair, count in cooccurrence.most_common(200)
            if pair[0] in node_ids and pair[1] in node_ids
        ]

        return {"nodes": nodes, "edges": edges}

    def get_author_network(self) -> dict:
        """Build co-authorship network."""
        analyses = self.store.load_all_analyses()

        author_pubs: dict[str, list] = defaultdict(list)
        coauthorship: dict[tuple, int] = Counter()

        for a in analyses:
            authors = [au.lower() for au in a.classification.authors]
            for au in authors:
                author_pubs[au].append(a.classification.title)
            for i, a1 in enumerate(authors):
                for a2 in authors[i + 1:]:
                    pair = tuple(sorted([a1, a2]))
                    coauthorship[pair] += 1

        nodes = [
            {"id": au, "label": au, "size": len(pubs)}
            for au, pubs in sorted(
                author_pubs.items(), key=lambda x: -len(x[1])
            )[:100]
        ]
        node_ids = {n["id"] for n in nodes}

        edges = [
            {"source": pair[0], "target": pair[1], "weight": count}
            for pair, count in coauthorship.items()
            if pair[0] in node_ids and pair[1] in node_ids
        ]

        return {"nodes": nodes, "edges": edges}

    def get_concept_map(self) -> dict:
        """Build hierarchical concept map from ontology data."""
        analyses = self.store.load_all_analyses()

        concepts = {}
        relations = []

        for a in analyses:
            for c in a.ontology.concepts:
                name = c.get("name", "").lower()
                if name:
                    concepts[name] = {
                        "id": name,
                        "label": c.get("name", ""),
                        "definition": c.get("definition", ""),
                        "domain": c.get("domain", ""),
                    }

            for r in a.ontology.concept_relationships:
                relations.append({
                    "source": r.get("from", "").lower(),
                    "target": r.get("to", "").lower(),
                    "relation": r.get("relation", "related"),
                })

        return {
            "nodes": list(concepts.values()),
            "edges": [
                r for r in relations
                if r["source"] in concepts and r["target"] in concepts
            ],
        }

    def get_overview_stats(self) -> dict:
        """Get comprehensive statistics for the dashboard."""
        db_stats = self.store.get_stats()
        graph_stats = self.graph.get_stats()

        analyses = self.store.load_all_analyses()
        all_domains = Counter()
        all_topics = Counter()
        all_keywords = Counter()
        all_methods = Counter()

        for a in analyses:
            for d in a.classification.research_domains:
                all_domains[d] += 1
            for t in a.classification.topics:
                all_topics[t] += 1
            for k in a.classification.keywords:
                all_keywords[k] += 1
            for m in a.classification.methodologies:
                all_methods[m] += 1

        return {
            "database": db_stats,
            "graph": graph_stats,
            "top_domains": dict(all_domains.most_common(10)),
            "top_topics": dict(all_topics.most_common(15)),
            "top_keywords": dict(all_keywords.most_common(20)),
            "top_methodologies": dict(all_methods.most_common(10)),
        }


def _most_common(items: list, n: int = 5) -> list[str]:
    return [item for item, _ in Counter(items).most_common(n)]
