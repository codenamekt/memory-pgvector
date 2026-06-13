"""mcp_server.cli — `hexus-mcp` console script entry point.

Forked from andreab67/hermes-hexus (BSD-3-Clause).

Usage:
  hexus-mcp serve --transport stdio --dsn "dbname=... user=... host=..."
  hexus-mcp serve --transport http  --host 0.0.0.0 --port 8000 \\
      --dsn "..." --agent-identity intraday-trading
  hexus-mcp doctor --dsn "..."        # one-shot health check + exit

The serve command blocks. The doctor command is for ops smoke-testing
("did the schema apply, is the embedder reachable, how many rows") and
is a CI-friendly way to validate a deployment without bringing up the
long-lived server.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Optional

logger = logging.getLogger("mcp_server")


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dsn",
        default=os.environ.get("HEXUS_DSN", ""),
        help=(
            "Postgres DSN. e.g. 'dbname=hermes_memory user=hermes host=/var/run/postgresql'. "
            "Defaults to env HEXUS_DSN. Required for serve."
        ),
    )
    p.add_argument(
        "--agent-identity",
        default=os.environ.get("HEXUS_AGENT_IDENTITY", ""),
        help=(
            "Default agent_identity for tool calls that don't supply one. "
            "Defaults to env HEXUS_AGENT_IDENTITY, then 'default'."
        ),
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("HEXUS_LOG_LEVEL", "INFO"),
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        stream=sys.stderr,  # MCP stdio transport owns stdout
    )


def cmd_serve(args: argparse.Namespace) -> int:
    if not args.dsn:
        print(
            "ERROR: --dsn is required (or set HEXUS_DSN env var).",
            file=sys.stderr,
        )
        return 2

    if args.agent_identity:
        # Set the default for the whole process. Tools read this on each
        # call that doesn't pass an explicit agent_identity.
        os.environ["HEXUS_AGENT_IDENTITY"] = args.agent_identity

    _configure_logging(args.log_level)
    logger.info(
        "starting hexus-mcp transport=%s agent=%r dsn=%s",
        args.transport,
        args.agent_identity or "<default>",
        _redact_dsn(args.dsn),
    )

    # Imported lazily so `hexus-mcp --help` works without
    # the [mcp] extra installed.
    from .server import build_server

    mcp = build_server(args.dsn, name="hexus")

    if args.transport == "stdio":
        # FastMCP's run('stdio') uses sys.stdin / sys.stdout, which is
        # exactly what Claude Desktop / Cursor / etc. expect.
        mcp.run(transport="stdio")
    elif args.transport == "http":
        # FastMCP's run('streamable-http') honors the host/port set on
        # the FastMCP instance. Update them here in case the operator
        # overrode on the CLI.
        try:
            mcp.settings.host = args.host
            mcp.settings.port = args.port
        except AttributeError:  # pragma: no cover — FastMCP internals
            pass
        mcp.run(transport="streamable-http")
    else:  # pragma: no cover — argparse should prevent this
        print(f"ERROR: unknown transport {args.transport!r}", file=sys.stderr)
        return 2
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """One-shot health check: print a JSON status line, exit 0/1.

    Designed for docker healthchecks and CI smoke tests. Does NOT
    load the BERT model — that would inflate every healthcheck by
    ~1-2s and a couple hundred MB. The lazy `LocalBertEmbedder` only
    loads on the first `embed()` call, which doctor doesn't trigger.
    """
    if not args.dsn:
        print(
            "ERROR: --dsn is required (or set HEXUS_DSN env var).",
            file=sys.stderr,
        )
        return 2

    from hexus.store import MemoryStore
    from . import tools

    _configure_logging(args.log_level)
    store = MemoryStore(args.dsn)
    try:
        status = tools.memory_health(store, {})
    finally:
        store.close()

    print(json.dumps(status, indent=2))
    return 0 if status.get("status") == "ok" else 1


def _redact_dsn(dsn: str) -> str:
    """Replace the password in a DSN with '***' for log lines."""
    import re

    return re.sub(r"(password\s*=\s*)([^\s]+)", r"\1***", dsn, flags=re.IGNORECASE)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hexus-mcp",
        description=(
            "hexus MCP server. Exposes the same Postgres + "
            "hexus memory store the hermes-agent plugin uses to any "
            "MCP client. Multi-agent: each connected client picks an "
            "agent_identity (CLI flag or per-call argument) and shares "
            "the same process / model load with every other client."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="Start the MCP server (blocks).")
    _add_common_args(p_serve)
    p_serve.add_argument(
        "--transport",
        default=os.environ.get("HEXUS_TRANSPORT", "stdio"),
        choices=("stdio", "http"),
        help=(
            "MCP transport. 'stdio' (default) is for editor integration "
            "(Claude Desktop, Cursor). 'http' exposes a streamable-http "
            "endpoint on --host/--port for fleet use."
        ),
    )
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    p_doc = sub.add_parser(
        "doctor",
        help="One-shot health check (prints JSON, exits 0/1).",
    )
    _add_common_args(p_doc)
    p_doc.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
