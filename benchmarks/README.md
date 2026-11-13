# Benchmarks

Skeleton only. The benchmark harness arrives in milestone 4 and will measure:

- numerical accuracy of the generic scalar/SIMD kernels against the Python
  reference;
- generic SIMD kernel versus a directly specialized SIMD implementation
  (target: within 1.10x execution time);
- sequential, sender-pipeline and coroutine execution of identical numerical
  work (throughput, median/p95 batch latency, peak memory, allocation counts);
- the workload size at which parallel execution becomes beneficial, and the
  scheduling/bandwidth limits beyond it.

No benchmark code is added in milestone 1; see `docs/design.md` for the
measurement plan.
