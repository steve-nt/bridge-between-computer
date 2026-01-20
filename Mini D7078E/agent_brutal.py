# agent_brutal.py - MAXIMUM THROUGHPUT, NO LIMITS
import asyncio
import aiohttp
import time
import argparse
import sys

async def worker(session, target_url, duration, worker_id):
    """Fire requests as FAST as possible, no rate limiting"""
    end = time.time() + duration
    count = 0
    errors = 0
    start_time = time.time()
    
    while time.time() < end:
        try:
            async with session.get(target_url, timeout=2, allow_redirects=False) as resp:
                status = resp.status
                try:
                    await resp.text()
                except:
                    pass
        except:
            errors += 1
        
        count += 1
        
        # Log every 1000 requests
        if count % 1000 == 0:
            elapsed = time.time() - start_time
            rps_actual = count / elapsed if elapsed > 0 else 0
            print(f"[w{worker_id}] {count} reqs, {errors} err, {rps_actual:.0f} RPS")
    
    elapsed_total = time.time() - start_time
    final_rps = count / elapsed_total if elapsed_total > 0 else 0
    print(f"[w{worker_id}] FINAL: {count} reqs in {elapsed_total:.1f}s = {final_rps:.0f} RPS, {errors} errors")

async def main(target_url, workers, duration):
    # Maximum aggressive settings
    timeout = aiohttp.ClientTimeout(total=10, connect=1, sock_read=2)
    conn = aiohttp.TCPConnector(
        limit=0,
        limit_per_host=0,
        force_close=False  # Keep connections alive
    )
    
    print(f"Starting {workers} workers, duration {duration}s")
    print(f"Target: {target_url}")
    print()
    
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        tasks = [asyncio.create_task(worker(session, target_url, duration, i))
                 for i in range(workers)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Target URL")
    parser.add_argument("--workers", type=int, default=100, help="Number of workers")
    parser.add_argument("--duration", type=int, default=120, help="Duration in seconds")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.workers, args.duration))
