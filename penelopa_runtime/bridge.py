"""The only managed stdio subprocess. No external credential is inherited."""

import argparse
import json
import sys

from penelopa_runtime.broker import TransportError, decode_rpc, exchange, rpc_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", required=True)
    args = parser.parse_args()
    if not args.broker.startswith("http://127.0.0.1:"):
        raise ValueError("Bridge requires a loopback broker")
    for line in sys.stdin:
        message = None
        try:
            message = json.loads(line)
            _, _, raw = exchange(args.broker + "/mcp", "penelopa-local", message, timeout=300)
            response = decode_rpc(raw)
        except (ValueError, TransportError):
            response = rpc_error(
                message.get("id") if isinstance(message, dict) else None,
                "Scoped MCP bridge unavailable",
            )
        if response is not None:
            print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
