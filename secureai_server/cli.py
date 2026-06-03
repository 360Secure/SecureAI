from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(prog="secureai")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="Start the SecureAI API gateway")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8787, type=int)
    serve.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.command == "serve":
        uvicorn.run("secureai_server.app:app", host=args.host, port=args.port, reload=args.reload)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
