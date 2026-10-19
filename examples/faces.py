"""Face detection: boxes and landmarks only; nobody is identified.

    pip install 'horasis[face]'
    # download face_detection_yunet_2023mar.onnx from the OpenCV Zoo, then:
    python examples/faces.py group.jpg yunet.onnx
"""

from __future__ import annotations

import sys

from _common import demo_client, image_argument


def run(client, key, image):
    result = client.face.detect_faces(image, api_key=key)
    for face in result.faces:
        print(f"face {face.confidence.value:.2f}  nose at {(face.landmarks or {}).get('nose')}")
    return result


if __name__ == "__main__":
    from horasis.backend import YuNetFaceAdapter

    model = sys.argv[2] if len(sys.argv) > 2 else None  # else the HORASIS_YUNET_MODEL env var
    client, key = demo_client(backends=[YuNetFaceAdapter(model_path=model)])
    run(client, key, image_argument())
