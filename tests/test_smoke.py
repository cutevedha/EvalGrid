# Smoke Test - Minimal end-to-end sanity check for the CI pipeline
# Verifies the demo runner executes without errors and returns well-formed results

from pipelines.ci import run_demo


def test_demo_runs():
    """
    Run the full demo pipeline and assert basic structural correctness.

    Checks:
    - The pipeline returns at least one result
    - Every result dict contains the required 'passed' key
    """
    results = run_demo()
    assert len(results) > 0
    assert all("passed" in r for r in results)
