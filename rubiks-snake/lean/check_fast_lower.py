"""Freshly rebuild the project-owned lower-bound proof, one module at a time.

The pinned compiler and existing third-party dependencies are reused. No
project-owned build artifacts are copied. Run with the repository's Python
environment; the temporary build is removed on success or failure.
"""

import argparse
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parent
TARGET = "RubiksSnake.CapLowerBound"
NATIVE = {"RubiksSnakeComputation", "RubiksSnakePrunedComputation", "RubiksSnakeCapComputation"}


def project_modules(target):
    ordered, visiting, seen = [], set(), set()

    def visit(module):
        if module in visiting:
            raise RuntimeError(f"cyclic module imports at {module}")
        if module in seen:
            return
        path = ROOT / (module.replace(".", "/") + ".lean")
        if not path.is_file():
            if module.startswith("RubiksSnake"):
                raise FileNotFoundError(path)
            return
        visiting.add(module)
        for dependency in re.findall(r"^import\s+([A-Za-z0-9_.]+)\s*$", path.read_text(), re.MULTILINE):
            visit(dependency)
        visiting.remove(module)
        seen.add(module)
        ordered.append(module)

    visit(target)
    return ordered


def run(output, seconds, target=TARGET):
    modules = project_modules(target)
    forbidden = {"RubiksSnake.SlabCountCertificate", "RubiksSnake.ExtendedSlabCertificate"}
    if forbidden.intersection(modules) or any("UpperBound" in module for module in modules):
        raise RuntimeError("the fast proof imports an expensive unrelated certificate")
    packages = ROOT / ".lake/packages"
    if not packages.is_dir():
        raise FileNotFoundError("existing Lake dependencies are required")
    env = dict(os.environ, LEAN_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    records = []
    with tempfile.TemporaryDirectory(prefix="rubiks-snake-fast-lower-") as directory:
        work = Path(directory)
        for name in ("lean-toolchain", "lakefile.toml", "lake-manifest.json"):
            shutil.copy2(ROOT / name, work / name)
        config = work / "lakefile.toml"
        config.write_text('moreLeanArgs = ["-j1", "-M4096"]\n' + config.read_text())
        for module in modules:
            relative = Path(module.replace(".", "/") + ".lean")
            (work / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, work / relative)
        (work / ".lake").mkdir()
        (work / ".lake/packages").symlink_to(packages, target_is_directory=True)
        start = time.monotonic()
        env["LEAN_PATH"] = subprocess.check_output(
            ["lake", "env", "printenv", "LEAN_PATH"], cwd=work, env=env, text=True
        ).strip()
        for module in modules:
            remaining = seconds - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError(f"fresh proof exceeded {seconds} seconds before {module}")
            if module in NATIVE:
                command = ["lake", "--no-cache", "build", module + ":shared"]
            else:
                relative = Path(module.replace(".", "/") + ".lean")
                olean_target = work / ".lake/build/lib/lean" / relative.with_suffix(".olean")
                olean_target.parent.mkdir(parents=True, exist_ok=True)
                libraries = [
                    library
                    for native in modules
                    if native in NATIVE
                    for library in sorted((work / ".lake/build/lib").glob(f"*_{native}.so"))
                ]
                command = ["lean", str(work / relative), "-o", str(olean_target), "-j1", "-M4096"]
                command.extend("--plugin=" + str(library) for library in libraries)
            command_start = time.monotonic()
            process = subprocess.Popen(
                command,
                cwd=work,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                log, _ = process.communicate(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.communicate()
                raise
            elapsed = time.monotonic() - command_start
            print(f"{module}: {elapsed:.3f}s", flush=True)
            if process.returncode:
                raise RuntimeError(log)
            records.append({"module": module, "seconds": elapsed, "output": log})
        elapsed = time.monotonic() - start
    report = {
        "target": target,
        "fresh_project_modules": len(modules),
        "reused": "pinned Lean compiler and cached third-party dependencies only",
        "seconds": elapsed,
        "limit_seconds": seconds,
        "under_limit": elapsed < seconds,
        "max_child_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "lean_heap_limit_mib": 4096,
        "serial_modules": True,
        "modules": records,
    }
    if output:
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "modules"}))
    if elapsed >= seconds:
        raise TimeoutError(f"fresh proof took {elapsed:.3f} seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--target", default=TARGET)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    run(args.output, args.seconds, args.target)


if __name__ == "__main__":
    main()
