"""The job runner must refuse anything not in the registry.

This is the whole security model: the cluster operator audits ops/jobs.yaml once
instead of reviewing every request, which only holds if unregistered jobs and
malformed parameters cannot execute.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "ops" / "run_job.py"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        capture_output=True, text=True, cwd=REPO, check=False,
    )


class JobRunnerRefusalTests(unittest.TestCase):
    def test_unregistered_job_is_refused(self) -> None:
        r = run("--job", "definitely_not_a_job", "--params", "{}")
        self.assertEqual(r.returncode, 2)
        self.assertIn("not a registered job", r.stderr)

    def test_path_traversal_is_refused(self) -> None:
        r = run("--job", "smoke", "--params",
                '{"raw_dir":"../../etc","out_dir":"/tmp/x"}')
        self.assertEqual(r.returncode, 2)
        self.assertIn("unsafe path", r.stderr)

    def test_shell_metacharacters_are_refused(self) -> None:
        for bad in ("/tmp/a; rm -rf /", "/tmp/a | sh", "/tmp/a && curl x", "/tmp/a`id`"):
            r = run("--job", "smoke", "--params",
                    '{"raw_dir":"%s","out_dir":"/tmp/x"}' % bad)
            self.assertEqual(r.returncode, 2, bad)
            self.assertIn("unsafe path", r.stderr, bad)

    def test_out_of_range_parameter_is_refused(self) -> None:
        r = run("--job", "extract_c18", "--params",
                '{"raw_dir":"/tmp/a","out_dir":"/tmp/b","workers":9999}')
        self.assertEqual(r.returncode, 2)
        self.assertIn("above max", r.stderr)

    def test_unknown_parameter_is_refused(self) -> None:
        r = run("--job", "smoke", "--params",
                '{"raw_dir":"/tmp/a","out_dir":"/tmp/b","extra_flag":"--danger"}')
        self.assertEqual(r.returncode, 2)
        self.assertIn("unknown parameter", r.stderr)

    def test_missing_required_parameter_is_refused(self) -> None:
        r = run("--job", "extract_c18", "--params", '{"raw_dir":"/tmp/a"}')
        self.assertEqual(r.returncode, 2)
        self.assertIn("missing required parameter", r.stderr)

    def test_registry_is_exactly_the_allowlisted_jobs(self) -> None:
        """The registry is the trust boundary; nothing may creep into it."""
        import yaml

        reg = yaml.safe_load((REPO / "ops" / "jobs.yaml").read_text())["jobs"]
        self.assertEqual(sorted(reg), ["extract_c18", "seal_holdout", "smoke"])

    def test_worker_ceiling_is_64(self) -> None:
        import yaml

        reg = yaml.safe_load((REPO / "ops" / "jobs.yaml").read_text())["jobs"]
        for name, job in reg.items():
            spec = (job.get("params") or {}).get("workers")
            if spec is not None:
                self.assertEqual(spec["max"], 64, name)

    def test_valid_request_resolves_argv_from_registry(self) -> None:
        r = run("--job", "seal_holdout", "--params", '{"out":"/tmp/h.json"}', "--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr)
        # argv is built from the registry template, never supplied by the request
        self.assertIn("scripts/seal_holdout.py", r.stdout)
        self.assertIn("--sealed-frac", r.stdout)


if __name__ == "__main__":
    unittest.main()
