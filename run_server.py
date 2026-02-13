#!/usr/bin/env python3
"""
Library Clerk - Visualization & Interaction Server

Start the web server to explore your knowledge graph.
Reads from the persisted SQLite database and graph data,
so no LLM inference is needed.

Usage:
    python run_server.py                        # Default config
    python run_server.py --port 8080            # Custom port
    python run_server.py --config my.yaml       # Custom config
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

from web.app import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("library_clerk.server")


def main():
    parser = argparse.ArgumentParser(
        description="Library Clerk - Visualization Server"
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--port", type=int, help="Override server port")
    parser.add_argument("--host", help="Override server host")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    server_cfg = config.get("server", {})
    host = args.host or server_cfg.get("host", "0.0.0.0")
    port = args.port or server_cfg.get("port", 5000)
    debug = args.debug or server_cfg.get("debug", False)

    # Verify persistence database exists
    db_path = config.get("persistence", {}).get("database", "library_clerk.db")
    if not Path(db_path).exists():
        logger.warning(
            f"Database {db_path} not found. Run the pipeline first:\n"
            f"  python run_pipeline.py"
        )

    app = create_app(config)

    print(f"\n{'='*50}")
    print(f"Library Clerk - Knowledge Graph Explorer")
    print(f"{'='*50}")
    print(f"  Server: http://{host}:{port}")
    print(f"  Database: {db_path}")
    print(f"{'='*50}\n")

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
