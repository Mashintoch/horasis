"""Pose estimation: body keypoints for each person.

    python examples/pose.py people.jpg
"""

from __future__ import annotations

from _common import demo_client, image_argument


def run(client, key, image, out="poses.png"):
    result = client.pose.estimate(image, api_key=key)
    for person in result.poses:
        visible = [k.name for k in person.keypoints if k.confidence.value > 0.5]
        print(f"person {person.confidence.value:.2f}: {len(visible)} visible keypoints")
    result.draw(image).save(out)
    return result


if __name__ == "__main__":
    client, key = demo_client()
    run(client, key, image_argument())
