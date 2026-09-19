"""Run the bundled assertion tests even when pytest is unavailable.

Kaggle normally includes pytest, but this fallback makes validation possible in
offline Anaconda environments using only the scientific dependencies required by
the project itself.
"""
from __future__ import annotations

import inspect
import runpy
import sys
import tempfile
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    namespace = runpy.run_path(str(root / "tests" / "test_core.py"))
    tests = sorted((name, obj) for name, obj in namespace.items()
                   if name.startswith("test_") and callable(obj))
    failures = []
    with tempfile.TemporaryDirectory(prefix="energy-opt-tests-") as tmp:
        tmp_path = Path(tmp)
        for name, test in tests:
            try:
                params = inspect.signature(test).parameters
                if params:
                    if set(params) != {"tmp_path"}:
                        raise TypeError(f"Unsupported test fixtures: {sorted(params)}")
                    test(tmp_path)
                else:
                    test()
                print(f"PASS  {name}")
            except Exception as exc:  # show every failed assertion in one run
                failures.append((name, exc))
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
