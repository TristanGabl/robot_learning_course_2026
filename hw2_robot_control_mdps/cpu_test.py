import time

def cpu_bench():
    n = 10_000_000
    s = time.perf_counter()
    t = 0
    for i in range(n):
        t += i * i
    return time.perf_counter() - s

dt = cpu_bench()
print(f"{10_000_000/dt:,.0f} ops/sec  ({dt:.2f}s)")
