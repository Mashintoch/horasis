"""Content moderation: per-category scores and a flagged verdict.

    pip install 'horasis[moderation]'
    python examples/moderate.py upload.jpg

Thresholds come from Config (``moderation_threshold`` / ``moderation_thresholds``).
"""

from __future__ import annotations

from _common import demo_client, image_argument


def run(client, key, image):
    result = client.moderation.moderate(image, api_key=key)
    for label in result.labels:
        flag = "  FLAGGED" if label.flagged else ""
        print(f"{label.category:<12} {label.confidence.value:.3f}{flag}")
    print("verdict:", "reject" if result.is_flagged else "ok")
    return result


if __name__ == "__main__":
    from horasis import Config
    from horasis.backend import TransformersModerationAdapter

    config = Config.default().model_copy(update={"moderation_thresholds": {"nsfw": 0.7}})
    client, key = demo_client(config=config, backends=[TransformersModerationAdapter()])
    run(client, key, image_argument())
