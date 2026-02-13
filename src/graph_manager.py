"""
Graph database manager with dual backend support.

Backends:
  - NetworkX: Lightweight, file-based, no external server required (default)
  - Neo4j: Full-featured graph database for production use

Both backends expose the same interface for building and querying the
publication knowledge graph.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import networkx as nx

logger = logging.getLogger(__name__)


class GraphBackend(ABC):
    """Abstract interface for graph database backends."""

    @abstractmethod
    def add_node(self, node_id: str, node_type: str, properties: dict) -> None:
        ...

    @abstractmethod
    def add_edge(
        self, source: str, target: str, relation: str,
        weight: float = 1.0, properties: dict | None = None,
    ) -> None:
        ...

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[dict]:
        ...

    @abstractmethod
    def get_neighbors(self, node_id: str, relation: str | None = None) -> list[dict]:
        ...

    @abstractmethod
    def get_all_nodes(self, node_type: str | None = None) -> list[dict]:
        ...

    @abstractmethod
    def get_all_edges(self, relation: str | None = None) -> list[dict]:
        ...

    @abstractmethod
    def search_nodes(self, query: str, node_type: str | None = None) -> list[dict]:
        ...

    @abstractmethod
    def get_subgraph(self, node_id: str, depth: int = 2) -> dict:
        ...

    @abstractmethod
    def get_stats(self) -> dict:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def export_json(self) -> dict:
        ...

    @abstractmethod
    def import_json(self, data: dict) -> None:
        ...


class NetworkXBackend(GraphBackend):
    """Lightweight graph backend using NetworkX. Persists to JSON."""

    def __init__(self, persist_path: str | None = None):
        self.graph = nx.MultiDiGraph()
        self.persist_path = persist_path
        if persist_path and Path(persist_path).exists():
            self._load()

    def add_node(self, node_id: str, node_type: str, properties: dict) -> None:
        self.graph.add_node(
            node_id, node_type=node_type, **properties
        )
        self._save()

    def add_edge(
        self, source: str, target: str, relation: str,
        weight: float = 1.0, properties: dict | None = None,
    ) -> None:
        props = properties or {}
        self.graph.add_edge(
            source, target,
            relation=relation, weight=weight, **props,
        )
        self._save()

    def get_node(self, node_id: str) -> Optional[dict]:
        if node_id not in self.graph:
            return None
        data = dict(self.graph.nodes[node_id])
        data["id"] = node_id
        return data

    def get_neighbors(self, node_id: str, relation: str | None = None) -> list[dict]:
        if node_id not in self.graph:
            return []
        results = []
        for _, target, data in self.graph.edges(node_id, data=True):
            if relation and data.get("relation") != relation:
                continue
            node_data = dict(self.graph.nodes[target])
            node_data["id"] = target
            node_data["_edge_relation"] = data.get("relation")
            node_data["_edge_weight"] = data.get("weight", 1.0)
            results.append(node_data)
        # Also check incoming edges
        for source, _, data in self.graph.in_edges(node_id, data=True):
            if relation and data.get("relation") != relation:
                continue
            node_data = dict(self.graph.nodes[source])
            node_data["id"] = source
            node_data["_edge_relation"] = data.get("relation")
            node_data["_edge_weight"] = data.get("weight", 1.0)
            results.append(node_data)
        return results

    def get_all_nodes(self, node_type: str | None = None) -> list[dict]:
        results = []
        for nid, data in self.graph.nodes(data=True):
            if node_type and data.get("node_type") != node_type:
                continue
            entry = dict(data)
            entry["id"] = nid
            results.append(entry)
        return results

    def get_all_edges(self, relation: str | None = None) -> list[dict]:
        results = []
        for u, v, data in self.graph.edges(data=True):
            if relation and data.get("relation") != relation:
                continue
            entry = dict(data)
            entry["source"] = u
            entry["target"] = v
            results.append(entry)
        return results

    def search_nodes(self, query: str, node_type: str | None = None) -> list[dict]:
        query_lower = query.lower()
        results = []
        for nid, data in self.graph.nodes(data=True):
            if node_type and data.get("node_type") != node_type:
                continue
            searchable = " ".join(
                str(v) for v in [nid, data.get("label", ""), data.get("title", "")]
            ).lower()
            if query_lower in searchable:
                entry = dict(data)
                entry["id"] = nid
                results.append(entry)
        return results

    def get_subgraph(self, node_id: str, depth: int = 2) -> dict:
        if node_id not in self.graph:
            return {"nodes": [], "edges": []}

        visited = set()
        frontier = {node_id}
        for _ in range(depth):
            next_frontier = set()
            for nid in frontier:
                visited.add(nid)
                for _, target, _ in self.graph.edges(nid, data=True):
                    if target not in visited:
                        next_frontier.add(target)
                for source, _, _ in self.graph.in_edges(nid, data=True):
                    if source not in visited:
                        next_frontier.add(source)
            frontier = next_frontier
        visited |= frontier

        nodes = []
        for nid in visited:
            if nid in self.graph:
                data = dict(self.graph.nodes[nid])
                data["id"] = nid
                nodes.append(data)

        edges = []
        for u, v, data in self.graph.edges(data=True):
            if u in visited and v in visited:
                entry = dict(data)
                entry["source"] = u
                entry["target"] = v
                edges.append(entry)

        return {"nodes": nodes, "edges": edges}

    def get_stats(self) -> dict:
        node_types = {}
        for _, data in self.graph.nodes(data=True):
            nt = data.get("node_type", "unknown")
            node_types[nt] = node_types.get(nt, 0) + 1

        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            et = data.get("relation", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "is_connected": nx.is_weakly_connected(self.graph)
            if self.graph.number_of_nodes() > 0
            else False,
            "components": nx.number_weakly_connected_components(self.graph)
            if self.graph.number_of_nodes() > 0
            else 0,
        }

    def clear(self) -> None:
        self.graph.clear()
        self._save()

    def export_json(self) -> dict:
        """Export full graph as JSON-serializable dict."""
        nodes = []
        for nid, data in self.graph.nodes(data=True):
            entry = {k: v for k, v in data.items() if _is_json_serializable(v)}
            entry["id"] = nid
            nodes.append(entry)

        edges = []
        for u, v, data in self.graph.edges(data=True):
            entry = {k: v for k, v in data.items() if _is_json_serializable(v)}
            entry["source"] = u
            entry["target"] = v
            edges.append(entry)

        return {"nodes": nodes, "edges": edges}

    def import_json(self, data: dict) -> None:
        """Import graph from JSON dict."""
        self.graph.clear()
        for node in data.get("nodes", []):
            nid = node.pop("id")
            self.graph.add_node(nid, **node)
        for edge in data.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            self.graph.add_edge(source, target, **edge)
        self._save()

    def _save(self):
        if not self.persist_path:
            return
        data = self.export_json()
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            self.import_json(data)
            logger.info(
                f"Loaded graph: {self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} edges"
            )
        except Exception as e:
            logger.warning(f"Failed to load graph from {self.persist_path}: {e}")


class Neo4jBackend(GraphBackend):
    """Neo4j graph database backend."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self._verify_connection()

    def _verify_connection(self):
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise

    def _run(self, query: str, **params) -> list[dict]:
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            return [dict(record) for record in result]

    def add_node(self, node_id: str, node_type: str, properties: dict) -> None:
        props = {k: v for k, v in properties.items() if _is_json_serializable(v)}
        props["id"] = node_id
        self._run(
            f"MERGE (n:{_safe_label(node_type)} {{id: $id}}) SET n += $props",
            id=node_id, props=props,
        )

    def add_edge(
        self, source: str, target: str, relation: str,
        weight: float = 1.0, properties: dict | None = None,
    ) -> None:
        props = {k: v for k, v in (properties or {}).items()
                 if _is_json_serializable(v)}
        props["weight"] = weight
        rel = _safe_label(relation)
        self._run(
            f"""MATCH (a {{id: $source}}), (b {{id: $target}})
            MERGE (a)-[r:{rel}]->(b)
            SET r += $props""",
            source=source, target=target, props=props,
        )

    def get_node(self, node_id: str) -> Optional[dict]:
        results = self._run(
            "MATCH (n {id: $id}) RETURN n, labels(n) as labels",
            id=node_id,
        )
        if not results:
            return None
        record = results[0]
        data = dict(record["n"])
        data["node_type"] = record["labels"][0] if record["labels"] else "Unknown"
        return data

    def get_neighbors(self, node_id: str, relation: str | None = None) -> list[dict]:
        if relation:
            rel_clause = f":{_safe_label(relation)}"
        else:
            rel_clause = ""
        results = self._run(
            f"""MATCH (a {{id: $id}})-[r{rel_clause}]-(b)
            RETURN b, labels(b) as labels, type(r) as rel, r.weight as weight""",
            id=node_id,
        )
        neighbors = []
        for r in results:
            data = dict(r["b"])
            data["node_type"] = r["labels"][0] if r["labels"] else "Unknown"
            data["_edge_relation"] = r["rel"]
            data["_edge_weight"] = r.get("weight", 1.0)
            neighbors.append(data)
        return neighbors

    def get_all_nodes(self, node_type: str | None = None) -> list[dict]:
        if node_type:
            results = self._run(
                f"MATCH (n:{_safe_label(node_type)}) RETURN n, labels(n) as labels"
            )
        else:
            results = self._run("MATCH (n) RETURN n, labels(n) as labels")
        nodes = []
        for r in results:
            data = dict(r["n"])
            data["node_type"] = r["labels"][0] if r["labels"] else "Unknown"
            nodes.append(data)
        return nodes

    def get_all_edges(self, relation: str | None = None) -> list[dict]:
        if relation:
            rel_clause = f":{_safe_label(relation)}"
        else:
            rel_clause = ""
        results = self._run(
            f"""MATCH (a)-[r{rel_clause}]->(b)
            RETURN a.id as source, b.id as target, type(r) as relation,
                   r.weight as weight, properties(r) as props"""
        )
        edges = []
        for r in results:
            entry = dict(r.get("props", {}))
            entry["source"] = r["source"]
            entry["target"] = r["target"]
            entry["relation"] = r["relation"]
            entry["weight"] = r.get("weight", 1.0)
            edges.append(entry)
        return edges

    def search_nodes(self, query: str, node_type: str | None = None) -> list[dict]:
        if node_type:
            results = self._run(
                f"""MATCH (n:{_safe_label(node_type)})
                WHERE toLower(n.id) CONTAINS toLower($q)
                   OR toLower(n.label) CONTAINS toLower($q)
                   OR toLower(n.title) CONTAINS toLower($q)
                RETURN n, labels(n) as labels LIMIT 50""",
                q=query,
            )
        else:
            results = self._run(
                """MATCH (n)
                WHERE toLower(n.id) CONTAINS toLower($q)
                   OR toLower(n.label) CONTAINS toLower($q)
                   OR toLower(n.title) CONTAINS toLower($q)
                RETURN n, labels(n) as labels LIMIT 50""",
                q=query,
            )
        nodes = []
        for r in results:
            data = dict(r["n"])
            data["node_type"] = r["labels"][0] if r["labels"] else "Unknown"
            nodes.append(data)
        return nodes

    def get_subgraph(self, node_id: str, depth: int = 2) -> dict:
        results = self._run(
            f"""MATCH path = (start {{id: $id}})-[*1..{depth}]-(end)
            WITH nodes(path) as ns, relationships(path) as rs
            UNWIND ns as n
            WITH COLLECT(DISTINCT n) as nodes, rs
            UNWIND rs as r
            RETURN nodes, COLLECT(DISTINCT r) as rels""",
            id=node_id,
        )
        # Fallback: return just the node
        if not results:
            node = self.get_node(node_id)
            return {"nodes": [node] if node else [], "edges": []}

        nodes = []
        edges = []
        for r in results:
            for n in r.get("nodes", []):
                data = dict(n)
                nodes.append(data)
            for rel in r.get("rels", []):
                edges.append({
                    "source": rel.start_node["id"],
                    "target": rel.end_node["id"],
                    "relation": rel.type,
                    "weight": rel.get("weight", 1.0),
                })
        return {"nodes": nodes, "edges": edges}

    def get_stats(self) -> dict:
        node_count = self._run("MATCH (n) RETURN count(n) as c")[0]["c"]
        edge_count = self._run("MATCH ()-[r]->() RETURN count(r) as c")[0]["c"]
        node_types = {}
        for r in self._run("MATCH (n) RETURN labels(n)[0] as label, count(*) as c"):
            node_types[r["label"]] = r["c"]
        edge_types = {}
        for r in self._run("MATCH ()-[r]->() RETURN type(r) as t, count(*) as c"):
            edge_types[r["t"]] = r["c"]

        return {
            "total_nodes": node_count,
            "total_edges": edge_count,
            "node_types": node_types,
            "edge_types": edge_types,
        }

    def clear(self) -> None:
        self._run("MATCH (n) DETACH DELETE n")

    def export_json(self) -> dict:
        return {
            "nodes": self.get_all_nodes(),
            "edges": self.get_all_edges(),
        }

    def import_json(self, data: dict) -> None:
        self.clear()
        for node in data.get("nodes", []):
            nid = node.get("id", "")
            ntype = node.get("node_type", "Entity")
            props = {k: v for k, v in node.items() if k not in ("id", "node_type")}
            self.add_node(nid, ntype, props)
        for edge in data.get("edges", []):
            self.add_edge(
                source=edge["source"],
                target=edge["target"],
                relation=edge.get("relation", "RELATED"),
                weight=edge.get("weight", 1.0),
                properties={
                    k: v for k, v in edge.items()
                    if k not in ("source", "target", "relation", "weight")
                },
            )

    def close(self):
        self.driver.close()


def _safe_label(label: str) -> str:
    """Sanitize a string for use as a Neo4j label or relationship type."""
    import re
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", label)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe or "Entity"


def _is_json_serializable(value: Any) -> bool:
    """Check if a value can be JSON-serialized."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_serializable(v) for v in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _is_json_serializable(v)
            for k, v in value.items()
        )
    return False


def create_backend(config: dict) -> GraphBackend:
    """Factory: create graph backend from config dict."""
    backend_type = config.get("backend", "networkx")

    if backend_type == "neo4j":
        neo4j_cfg = config.get("neo4j", {})
        return Neo4jBackend(
            uri=neo4j_cfg.get("uri", "bolt://localhost:7687"),
            user=neo4j_cfg.get("user", "neo4j"),
            password=neo4j_cfg.get("password", "password"),
            database=neo4j_cfg.get("database", "neo4j"),
        )
    else:
        return NetworkXBackend(persist_path="graph_data.json")
