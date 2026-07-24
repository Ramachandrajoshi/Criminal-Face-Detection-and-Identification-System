"""
Dedicated thread pool for offloading blocking DeepFace/OpenCV inference calls
off the FastAPI event loop.

Why a dedicated pool instead of ``asyncio.to_thread``
-------------------------------------------------------
``asyncio.to_thread`` uses the event loop's *default* executor
(``ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4))``), sized for
short I/O-bound work. Face detection/embedding is GPU-bound and shares a
single card (see ``docker-compose.yml`` GPU config) — concurrency needs to be
capped independently so we don't queue more simultaneous inference calls than
the GPU can usefully serve. A dedicated, explicitly-sized pool makes that cap
a single tunable (``INFERENCE_POOL_SIZE``) instead of an implicit default.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, TypeVar

from app.core.config import settings

T = TypeVar("T")

_executor = ThreadPoolExecutor(
    max_workers=settings.inference_pool_size,
    thread_name_prefix="inference",
)


def get_executor() -> ThreadPoolExecutor:
    """Return the shared inference thread pool (for direct ``run_in_executor`` use)."""
    return _executor


async def run_inference(fn: Callable[..., T], *args, **kwargs) -> T:
    """Run a blocking callable on the shared inference pool and await its result."""
    loop = asyncio.get_running_loop()
    bound = partial(fn, *args, **kwargs) if kwargs else fn
    if kwargs:
        return await loop.run_in_executor(_executor, bound)
    return await loop.run_in_executor(_executor, fn, *args)
