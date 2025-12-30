# agent_ultra.py - aggressive load generator with removed bottlenecks
import asyncio
import aiohttp
import time
import argparse
import sys

async def worker(session, target_url, rps, duration, worker_id):
    interval = 1.0 / rps if rps > 0 else 1.0
    end = time.time() + duration
    count = 0
    errors = 0
    start_time = time.time()
    
    while time.time() < end:
        req_start = time.time()
        try:
            async with session.get(target_url, timeout=3, allow_redirects=False) as resp:
                status = resp.status
                try:
                    _ = await resp.text()
                except:
                    pass
        except asyncio.TimeoutError:
            errors += 1
        except Exception as e:
            errors += 1
        
        count += 1
        
        # Log periodically
        if count % max(1, int(rps * 5)) == 0:
            elapsed = time.time() - start_time
            rps_actual = count / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\r[w{worker_id}] {count} reqs, {errors} err, {rps_actual:.1f} RPS")
            sys.stdout.flush()
        
        # Rate-limit
        elapsed = time.time() - req_start
        sleep = interval - elapsed
        if sleep > 0:
            await asyncio.sleep(sleep)
    
    elapsed_total = time.time() - start_time
    final_rps = count / elapsed_total if elapsed_total > 0 else 0
    print(f"\n[w{worker_id}] DONE: {count} requests in {elapsed_total:.1f}s ({final_rps:.1f} RPS), {errors} errors")

async def main(target_url, workers, rps_per_worker, duration, mode):
    # AGGRESSIVE MODE: Remove all connection limits
    if mode == "aggressive":
        # Unlimited connections, short timeout, force close
        timeout = aiohttp.ClientTimeout(total=5, connect=2)
        conn = aiohttp.TCPConnector(
            limit=0,  # NO connection limit
            limit_per_host=0,  # NO per-host limit
            force_close=True,  # Close after each request
            ttl_dns_cache=None  # Fresh DNS lookup each time
        )
    else:
        # Standard mode (fallback)
        timeout = aiohttp.ClientTimeout(total=5)
        conn = aiohttp.TCPConnector(limit_per_host=workers, force_close=True)
    
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        tasks = [asyncio.create_task(worker(session, target_url, rps_per_worker, duration, i))
                 for i in range(workers)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Target URL (ALB)")
    parser.add_argument("--workers", type=int, default=4, help="Number of coroutines")
    parser.add_argument("--rps", type=float, default=1.0, help="Requests per second per worker")
    parser.add_argument("--duration", type=int, default=120, help="Duration in seconds")
    parser.add_argument("--mode", default="standard", help="aggressive or standard")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.workers, args.rps, args.duration, args.mode))
