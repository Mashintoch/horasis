"""Async client: concurrent batch detection with per-item errors and usage.

    python examples/async_batch.py a.jpg b.jpg missing.jpg
"""

from __future__ import annotations

import asyncio
import sys

from _common import demo_client


async def run(client, key, images):
    from horasis import AsyncSkopos

    async with AsyncSkopos(client, max_workers=4) as aclient:
        items = await aclient.detection.detect_batch(images, api_key=key)
        for item in items:
            if item.ok:
                print(f"{item.source}: {len(item.result.detections)} objects")
            else:
                print(f"{item.source}: FAILED -- {item.error.message}")
        used = (await aclient.usage(key)).overall
        print(f"quota used: {used.used}/{used.limit}")
        return items


if __name__ == "__main__":
    client, key = demo_client()
    asyncio.run(run(client, key, sys.argv[1:]))
