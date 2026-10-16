"""Instance segmentation: outlines for every object.

    python examples/segment.py photo.jpg
"""

from __future__ import annotations

from _common import demo_client, image_argument


def run(client, key, image, out="segmented.png"):
    result = client.segmentation.segment(image, api_key=key)
    for seg in result.segments:
        print(f"{seg.label:<14} {seg.confidence.value:.2f}  {len(seg.polygon)} outline points")
    result.draw(image).save(out)
    return result


if __name__ == "__main__":
    client, key = demo_client()
    run(client, key, image_argument())
