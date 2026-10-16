"""Use a remote horasis server as just another backend.

    pip install 'horasis[remote]'
    python examples/remote_client.py photo.jpg http://127.0.0.1:8000 change-me
"""

from __future__ import annotations

import sys

from _common import demo_client


def run(client, key, image):
    result = client.detection.detect(image, api_key=key, backend="remote")
    print(f"{len(result.detections)} objects, answered by backend {result.backend!r}")
    return result


if __name__ == "__main__":
    from horasis.backend import RemoteAdapter

    image, url, server_key = sys.argv[1], sys.argv[2], sys.argv[3]
    client, key = demo_client(backends=[RemoteAdapter(endpoint_url=url, api_key=server_key)])
    run(client, key, image)
