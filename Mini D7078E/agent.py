# agent.py — asynchronous, rate-limited HTTP requester for lab-only use
import asyncio
import aiohttp
import time
import argparse
import random

async def worker(session, target_url, rps, duration, worker_id):
    interval = 1.0 / rps if rps > 0 else 1.0
    end = time.time() + duration
    count = 0
    while time.time() < end:
        start = time.time()
        try:
            async with session.get(target_url, timeout=10) as resp:
                status = resp.status
                _ = await resp.text()
        except Exception as e:
            status = 0
        count += 1
        # simple logging
        if count % max(1, int(rps)) == 0:
            print(f"[worker {worker_id}] req #{count} status={status}")
        # rate-limit
        elapsed = time.time() - start
        sleep = interval - elapsed
        if sleep > 0:
            await asyncio.sleep(sleep)
        else:
            # if overloaded, yield briefly
            await asyncio.sleep(0.001)
    print(f"[worker {worker_id}] finished, total requests: {count}")

async def main(target_url, workers, rps_per_worker, duration):
    timeout = aiohttp.ClientTimeout(total=20)
    conn = aiohttp.TCPConnector(limit_per_host=workers, force_close=True)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        tasks = [asyncio.create_task(worker(session, target_url, rps_per_worker, duration, i))
                 for i in range(workers)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Target URL (ALB)")
    parser.add_argument("--workers", type=int, default=4, help="Number of coroutines in agent")
    parser.add_argument("--rps", type=float, default=1.0, help="Requests per second per worker")
    parser.add_argument("--duration", type=int, default=120, help="Duration in seconds")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.workers, args.rps, args.duration))
