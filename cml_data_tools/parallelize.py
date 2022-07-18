"""Lightweight tools for running parallel processes"""
from concurrent.futures import as_completed, ProcessPoolExecutor, TimeoutError


def _drain_queue(q, timeout=None):
    try:
        for future in as_completed(q, timeout=timeout):
            yield future.result()
            q.remove(future)
    except TimeoutError:
        pass


def parallelize_func(func, data, *func_args, max_workers=8):
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = set()
        for df in data:
            fut = pool.submit(func, df, *func_args)
            futures.add(fut)
            if len(futures) > 100:
                yield from _drain_queue(futures, timeout=1)
        # Block until all futures arrive
        yield from _drain_queue(futures)
