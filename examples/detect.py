"""Object detection: print what was found and save an annotated copy.

    pip install 'horasis[ultralytics]'
    python examples/detect.py photo.jpg
"""

from __future__ import annotations

from _common import demo_client, image_argument


def run(client, key, image, out="detected.png"):
    result = client.detection.detect(image, api_key=key)
    for d in result.filter(min_conf=0.3).detections:
        print(f"{d.label:<14} {d.confidence.value:.2f}  {d.box.x_min:.0f},{d.box.y_min:.0f}")
    result.draw(image).save(out)  # draw() returns a PIL image and saves nothing itself
    return result


if __name__ == "__main__":
    client, key = demo_client()
    run(client, key, image_argument())
