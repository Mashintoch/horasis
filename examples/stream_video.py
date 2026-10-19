"""Stream detections from a video file or webcam. Each processed frame is billed.

    pip install 'horasis[video]'
    python examples/stream_video.py clip.mp4      # a file
    python examples/stream_video.py 0             # webcam 0
"""

from __future__ import annotations

import sys

from _common import demo_client


def run(client, key, source, every_n_frames=10, max_frames=50):
    count = 0
    for frame in client.detection.stream(
        source, api_key=key, every_n_frames=every_n_frames, max_frames=max_frames
    ):
        labels = sorted({d.label for d in frame.result.detections})
        print(f"t={frame.timestamp_s:6.2f}s frame {frame.frame_index:5d}: {labels}")
        count += 1
    return count


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    client, key = demo_client()
    run(client, key, int(arg) if arg.isdigit() else arg)
