"""OCR: read text, line by line, with confidences.

    pip install 'horasis[ocr]'
    python examples/ocr.py scan.png
"""

from __future__ import annotations

from _common import demo_client, image_argument


def run(client, key, image):
    result = client.ocr.extract_text(image, api_key=key)
    for line in result.lines:
        print(f"{line.confidence.value:.2f}  {line.text}")
    return result


if __name__ == "__main__":
    from horasis.backend import EasyOCRAdapter

    client, key = demo_client(backends=[EasyOCRAdapter(languages=["en"])])
    run(client, key, image_argument())
