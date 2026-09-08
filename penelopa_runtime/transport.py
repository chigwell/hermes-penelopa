"""Fixed HTTP worker; the supervisor kills/reaps it at the physical deadline.

Credentials arrive only on stdin, never argv, files or environment. This is
trusted runtime plumbing, not an executable capability exposed to Hermes.
"""

import base64
import json
import sys
import urllib.error
import urllib.request

from penelopa_runtime.diagnostics import provider_request_id_from_headers


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def main():
    value = json.load(sys.stdin)
    request = urllib.request.Request(
        value["url"],
        data=None if value["payload"] is None else json.dumps(value["payload"]).encode(),
        method=value["method"],
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **(value["headers"] or {}),
            "Authorization": f"Bearer {value['token']}",
        },
    )
    try:
        with urllib.request.build_opener(NoRedirect).open(
            request, timeout=value["timeout"]
        ) as response:
            result = {
                "status": response.status,
                "headers": dict(response.headers),
                "body": base64.b64encode(response.read()).decode("ascii"),
            }
    except urllib.error.HTTPError as error:
        # Do not read the error body or return arbitrary headers.  A status and
        # one allowlisted opaque provider request ID are sufficient to diagnose
        # an upstream rejection without persisting prompts, responses or tokens.
        result = {"error": f"http_{error.code}", "http_status": error.code}
        request_id = provider_request_id_from_headers(error.headers)
        if request_id is not None:
            result["provider_request_id"] = request_id
    except (OSError, urllib.error.URLError, TimeoutError):
        result = {"error": "transport_unavailable"}
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
