"""Image classification: the top labels for a picture.

    python examples/classify.py photo.jpg
"""

from __future__ import annotations

from _common import demo_client, image_argument


def run(client, key, image):
    result = client.classification.classify(image, api_key=key)
    for entry in result.entries:
        print(f"{entry.label:<24} {entry.confidence.value:.3f}")
    return result


if __name__ == "__main__":
    client, key = demo_client()
    run(client, key, image_argument())
