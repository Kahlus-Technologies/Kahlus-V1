import re
import shutil
import subprocess
import tarfile
import unittest
from pathlib import Path


def copy_repo_to_temp_git(tmp: str) -> Path:
    repo_root = Path.cwd()
    files = subprocess.run(
        ["git", "ls-files", "--cached"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()

    tmp_repo = Path(tmp) / "repo"
    tmp_repo.mkdir()
    for rel in files:
        source = repo_root / rel
        if not source.is_file():
            continue
        target = tmp_repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    subprocess.run(["git", "init", "-q"], cwd=tmp_repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_repo, check=True)
    subprocess.run(["git", "config", "user.name", "Bundle Test"], cwd=tmp_repo, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_repo, check=True)
    # A throwaway repo must not run background maintenance. `git commit` can fire
    # `gc --auto`, which writes .git/objects/info/packs and .git/info/refs after
    # the test body finishes -- racing TemporaryDirectory cleanup and failing it
    # with "Directory not empty". Observed intermittently (~40% of runs) on macOS.
    subprocess.run(["git", "config", "gc.auto", "0"], cwd=tmp_repo, check=True)
    subprocess.run(["git", "config", "maintenance.auto", "false"], cwd=tmp_repo, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "bundle test"], cwd=tmp_repo, check=True)
    return tmp_repo


def assert_runner_archive(testcase: unittest.TestCase, archive: Path, extract_root: Path) -> Path:
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
        try:
            tar.extractall(path=extract_root, filter="data")
        except TypeError:
            tar.extractall(path=extract_root)

    roots = {name.split("/", 1)[0] for name in names if name}
    testcase.assertEqual(len(roots), 1)
    root = roots.pop()

    required = {
        "BUNDLE_MANIFEST.txt",
        "BUNDLE_METADATA.txt",
        "COMMIT_HASH.txt",
        "Dockerfile.a100",
        "README.md",
        "README_AGENT_DEPLOY.md",
        "README_HANDOFF.md",
        "README_HANDOFF.md.in",
        "README_RUN.md",
        "SHA256SUMS",
        "configs/train/moabb_a100.yaml",
        "configs/train/moabb_a100_smoke.yaml",
        "configs/train/neurotwin_v1_a100.yaml",
        "configs/train/algonauts_real_stimulus_debug.yaml",
        "configs/train/algonauts_pair_operator_debug.yaml",
        "configs/train/algonauts_pair_operator_full.yaml",
        "configs/train/algonauts_pair_operator_ablation_array.yaml",
        "configs/train/synthetic_field_compiler_debug.yaml",
        "configs/train/algonauts_field_compiler_debug.yaml",
        "configs/train/moabb_field_compiler_debug.yaml",
        "configs/train/prepared_synthetic_debug.yaml",
        "docs/research/math_implementation_coverage.md",
        "environment-a100.yml",
        "pyproject.toml",
        "requirements/cluster-a100.txt",
        "scripts/lib/moabb_prepare_common.sh",
        "scripts/package_a100_evidence_bundle.sh",
        "scripts/package_a100_evidence_bundle.py",
        "scripts/package_runner_bundle.sh",
        "scripts/render_a100_handoff_readme.py",
        "scripts/prepare_moabb_benchmark.sh",
        "scripts/docker_a100_inner.sh",
        "scripts/docker_gpu_preflight.py",
        "scripts/run_docker_7gpu.sh",
        "scripts/run_docker_6gpu.sh",
        "scripts/run_full.sbatch",
        "scripts/run_full.sh",
        "scripts/run_smoke.sh",
        "scripts/slurm/_train_a100_inner.sh",
        "scripts/slurm/train_a100.sh",
        "scripts/train_a100_inner.sh",
        "src/neurotwin/data/__init__.py",
        "src/neurotwin/data/windows.py",
    }
    for rel in required:
        testcase.assertIn(f"{root}/{rel}", names)

    forbidden_parts = {
        ".context",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "graphify-out",
        "outputs",
        "runs",
        "tests",
    }
    forbidden_names = {"pw.txt", ".env"}
    forbidden_suffixes = (".ckpt", ".npy", ".npz", ".pt", ".pth", ".pyc", ".pem", ".key")
    for name in names:
        rel = name.split("/", 1)[1] if "/" in name else ""
        parts = Path(rel).parts
        testcase.assertFalse(any(part.startswith("._") for part in parts), name)
        testcase.assertFalse(forbidden_parts.intersection(parts), name)
        testcase.assertFalse(any(part.lower().startswith(".env.") for part in parts), name)
        testcase.assertFalse(forbidden_names.intersection(parts), name)
        testcase.assertFalse(rel.startswith("docs/paper/"), name)
        if rel != "docs/research/math_implementation_coverage.md":
            testcase.assertFalse(rel.startswith("docs/research/"), name)
        testcase.assertFalse(rel.endswith(forbidden_suffixes), name)

    bundle_root = extract_root / root
    manifest = (bundle_root / "BUNDLE_MANIFEST.txt").read_text(encoding="utf-8").splitlines()
    payload = sorted(
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file() and path.name not in {"BUNDLE_MANIFEST.txt", "SHA256SUMS"}
    )
    testcase.assertEqual(manifest, payload)

    checksum = subprocess.run(
        ["shasum", "-a", "256", "-c", "SHA256SUMS"],
        cwd=bundle_root,
        text=True,
        capture_output=True,
    )
    testcase.assertEqual(checksum.returncode, 0, checksum.stderr + checksum.stdout)
    readme = (bundle_root / "README_RUN.md").read_text(encoding="utf-8")
    documented_scripts = set(re.findall(r"(?<![\w/.-])(scripts/[A-Za-z0-9_./-]+(?:\.sh|\.sbatch))", readme))
    testcase.assertIn("scripts/slurm/train_a100.sh", documented_scripts)
    for rel in documented_scripts:
        testcase.assertIn(f"{root}/{rel}", names)
    return bundle_root
