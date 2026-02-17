"""
Flask web application for the Library Clerk visualization and interaction interface.

Provides:
  - Interactive 2D network graph (vis.js)
  - 3D force-directed point cloud (3d-force-graph)
  - Search and exploration interface
  - Publication detail views
  - Dashboard with statistics
  - REST API for graph data
"""

import json
import logging
from pathlib import Path

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

from src.graph_manager import create_backend, NetworkXBackend
from src.persistence import PersistenceStore
from src.query_engine import QueryEngine

logger = logging.getLogger(__name__)


def create_app(config: dict) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    CORS(app)

    # Initialize backends
    persistence_cfg = config.get("persistence", {})
    store = PersistenceStore(
        db_path=persistence_cfg.get("database", "library_clerk.db")
    )

    # Load persisted graph
    graph_cfg = config.get("graph", {})
    if graph_cfg.get("backend") == "neo4j":
        graph = create_backend(graph_cfg)
    else:
        graph = NetworkXBackend(persist_path="graph_data.json")
        saved = store.load_graph_state("full_graph")
        if saved:
            graph.import_json(saved)
            logger.info(
                f"Loaded graph: {graph.get_stats()['total_nodes']} nodes, "
                f"{graph.get_stats()['total_edges']} edges"
            )

    engine = QueryEngine(graph, store)

    # --- Page routes ---

    @app.route("/")
    def index():
        stats = engine.get_overview_stats()
        return render_template("index.html", stats=stats)

    # --- API routes ---

    @app.route("/api/graph")
    def api_graph():
        """Full graph data for visualization."""
        return jsonify(graph.export_json())

    @app.route("/api/graph/subgraph/<node_id>")
    def api_subgraph(node_id):
        """Subgraph around a specific node."""
        depth = request.args.get("depth", 2, type=int)
        return jsonify(graph.get_subgraph(node_id, depth=depth))

    @app.route("/api/search")
    def api_search():
        """Search the knowledge base."""
        query = request.args.get("q", "")
        node_type = request.args.get("type", None)
        if not query:
            return jsonify({"error": "Query parameter 'q' required"}), 400
        return jsonify(engine.search(query, node_type))

    @app.route("/api/publications")
    def api_publications():
        """List all publications."""
        return jsonify(store.get_publications_summary())

    @app.route("/api/publication/<file_hash>")
    def api_publication(file_hash):
        """Get full publication details."""
        detail = engine.get_publication_detail(file_hash)
        if not detail:
            return jsonify({"error": "Publication not found"}), 404
        return jsonify(detail)

    @app.route("/api/publication/<file_hash>/related")
    def api_related(file_hash):
        """Find related publications."""
        limit = request.args.get("limit", 10, type=int)
        return jsonify(engine.find_related(file_hash, limit))

    @app.route("/api/clusters")
    def api_clusters():
        """Get publication clusters by domain."""
        return jsonify(engine.get_clusters())

    @app.route("/api/timeline")
    def api_timeline():
        """Get research timeline."""
        return jsonify(engine.get_research_timeline())

    @app.route("/api/keywords")
    def api_keywords():
        """Get keyword co-occurrence network."""
        return jsonify(engine.get_keyword_network())

    @app.route("/api/authors")
    def api_authors():
        """Get co-authorship network."""
        return jsonify(engine.get_author_network())

    @app.route("/api/concepts")
    def api_concepts():
        """Get concept map."""
        return jsonify(engine.get_concept_map())

    @app.route("/api/stats")
    def api_stats():
        """Get overview statistics."""
        return jsonify(engine.get_overview_stats())

    @app.route("/api/node/<node_id>")
    def api_node(node_id):
        """Get a single node with its neighbors."""
        # Try the exact ID first, then resolve from focused-view label
        real_id = engine._resolve_graph_id(node_id)
        if not real_id:
            return jsonify({"error": "Node not found"}), 404
        node = graph.get_node(real_id)
        neighbors = graph.get_neighbors(real_id)
        return jsonify({"node": node, "neighbors": neighbors})

    @app.route("/api/node/<node_id>/publications")
    def api_node_publications(node_id):
        """Get all publications linked to a given node."""
        pubs = engine.get_publications_for_node(node_id)
        return jsonify(pubs)

    @app.route("/api/edge", methods=["POST"])
    def api_add_edge():
        """Manually add an edge between two nodes."""
        data = request.get_json(silent=True) or {}
        source = data.get("source", "").strip()
        target = data.get("target", "").strip()
        relation = data.get("relation", "").strip()

        if not source or not target or not relation:
            return jsonify({"error": "source, target, and relation are required"}), 400

        if not graph.get_node(source):
            return jsonify({"error": f"Source node '{source}' not found"}), 404
        if not graph.get_node(target):
            return jsonify({"error": f"Target node '{target}' not found"}), 404

        weight = float(data.get("weight", 0.8))
        graph.add_edge(source, target, relation.upper(), weight=weight)
        store.store_graph_state("full_graph", graph.export_json())

        logger.info(f"Manual edge added: {source} -[{relation.upper()}]-> {target}")
        return jsonify({"ok": True, "source": source, "target": target, "relation": relation.upper()})

    return app
