"""evalgrid.benchmarks: reproducible benchmarks comparing EvalGrid against other frameworks."""

from evalgrid.benchmarks.deepeval_comparison import (
    DeepEvalSimulator,
    benchmark_vs_deepeval,
    print_comparison,
)

__all__ = ["DeepEvalSimulator", "benchmark_vs_deepeval", "print_comparison"]
