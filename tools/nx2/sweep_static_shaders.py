#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Lower every extracted shader and separate captured replay from synthetic context."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

from shader_sweep import invoke, parse_cache, unpack_native_code, summarize


def digest(data):
    return hashlib.sha256(data).hexdigest()


def artifact_path(directory, relative):
    path = (directory / relative.replace("\\", "/")).resolve()
    if directory.resolve() not in path.parents:
        raise ValueError("program path escapes its extraction directory")
    return path


def output_path(path, protected_roots):
    path = path.resolve()
    if path.exists() or any(root.resolve() == path or root.resolve() in path.parents
                            for root in protected_roots):
        raise ValueError("output must be a new directory outside all read-only inputs")
    return path


def compute_format(code, control):
    if (not code or len(code) % 16 or len(control) < 104 or
            control[:8] != bytes.fromhex("34127698c9000000")):
        return False
    size, = struct.unpack_from("<I", control, 56)
    stage, = struct.unpack_from("<I", control, 84)
    return 0 < size <= len(code) and size % 16 == 0 and stage >> 16 == 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--read-only-root", action="append", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--spirv-val", default="spirv-val")
    parser.add_argument("--static-context", action="store_true",
                        help="without a cache match, derive driver context from the control section")
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout < 1 or not args.compiler.is_file():
        parser.error("positive jobs/timeout and a compiler executable are required")
    try:
        args.output = output_path(args.output, args.read_only_root +
                                  [path.parent for path in args.manifests] +
                                  ([args.cache_directory] if args.cache_directory else []))
    except ValueError as exc:
        parser.error(str(exc))
    validator = shutil.which(args.spirv_val)
    if not validator:
        parser.error("spirv-val is required")
    args.output.mkdir(parents=True)
    records = args.output / "records"
    records.mkdir(exist_ok=True)
    metadata = {"compiler": str(args.compiler.resolve()),
                "compiler_sha256": digest(args.compiler.read_bytes()),
                "runner_sha256": digest(Path(__file__).read_bytes()),
                "validator": validator, "validator_sha256": digest(Path(validator).read_bytes()),
                "manifests": [], "read_only_roots": [str(path.resolve()) for path in args.read_only_root],
                "cache_directory": str(args.cache_directory) if args.cache_directory else None,
                "cache_context_policy": "one compatible candidate per allocation; association remains inferred"}
    cached = defaultdict(list)
    cache_errors = []
    if args.cache_directory:
        for path in sorted(args.cache_directory.glob("*/vulkan.fork-v*.bin")):
            try:
                for environment in parse_cache(path.read_bytes()):
                    lowered = unpack_native_code(environment["code"])
                    if lowered is not None:
                        environment["source"] = str(path)
                        cached[(path.parent.name.lower(), environment["stage"], digest(lowered))].append(environment)
            except (ValueError, OSError) as exc:
                cache_errors.append({"source": str(path), "error": str(exc)})
    programs = {}
    archive_errors = []
    for manifest_path in args.manifests:
        data = manifest_path.read_bytes()
        metadata["manifests"].append({"path": str(manifest_path.resolve()), "sha256": digest(data)})
        manifest = json.loads(data)
        archive_errors.extend(manifest["errors"])
        for program in manifest["programs"]:
            identity = (program["title_id"], program["stage"], program["code_sha256"], program["control_sha256"])
            if identity not in programs:
                programs[identity] = {**program, "directory": manifest_path.parent}
            else:
                programs[identity]["origins"].extend(program["origins"])
    if not programs:
        archive_errors.append({"phase": "coverage", "error": "no extracted shader programs to sweep"})
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    def compile_one(program):
        identity = digest((program["title_id"] + program["stage"] + program["code_sha256"] +
                           program["control_sha256"]).encode())
        base = {"game": program["game"], "title_id": program["title_id"], "stage": program["stage"],
                "source": program["code"], "control": program["control"],
                "origins": program["origins"], "id": identity,
                "code_sha256": program["code_sha256"], "control_sha256": program["control_sha256"]}
        try:
            for field in ("code", "control"):
                path = artifact_path(program["directory"], program[field])
                data = path.read_bytes()
                if digest(data) != program[f"{field}_sha256"]:
                    raise ValueError(f"{field} SHA256 does not match extraction manifest")
                base["source" if field == "code" else field] = str(path)
        except (OSError, ValueError) as exc:
            return {**base, "status": "error", "phase": "input", "lowered": False,
                    "error": {"message": str(exc)}}
        fmt = program["format"]
        compiler_source = base["source"]
        if fmt is None:
            if base["stage"] != "compute":
                return {**base, "status": "error", "phase": "context", "lowered": False,
                        "error": {"message": f"unverified {base['stage']} bytecode envelope and launch metadata"}}
            if not compute_format(Path(base["source"]).read_bytes(), Path(base["control"]).read_bytes()):
                return {**base, "status": "error", "phase": "context", "lowered": False,
                        "error": {"message": "unverified compute bytecode envelope"}}
            code = Path(base["source"]).read_bytes()
            control = Path(base["control"]).read_bytes()
            code_size, = struct.unpack_from("<I", control, 56)
            bounded = records / f"{identity}.instructions"
            bounded.write_bytes(code[:code_size])
            compiler_source = str(bounded)
            base["archive_allocation_bytes"] = len(code)
            base["declared_code_bytes"] = code_size
            fmt = "instructions"
        lowered_path = records / f"{identity}.lowered"
        command = [str(args.compiler.resolve()), compiler_source, "--stage", base["stage"],
                   "--format", fmt, "--capture-extent"]
        lowered = invoke(command + ["--lower-only", str(lowered_path)], args.timeout)
        if lowered["status"] != "ok":
            row = {**base, **lowered, "lowered": False}
            offset = lowered.get("error", {}).get("offset")
            if isinstance(offset, int):
                data = Path(base["source"]).read_bytes()
                offset += 128 if fmt == "native-header" else 0
                if 0 <= offset <= len(data) - 16:
                    words = struct.unpack_from("<4I", data, offset)
                    row["opcode"] = f"0x{words[0] & 0xfff:03x}"
                    row["instruction"] = [f"{word:08x}" for word in words]
            return row
        matches = cached.get((base["title_id"].lower(), base["stage"], digest(lowered_path.read_bytes())), [])
        if fmt == "native-header":
            header = Path(base["source"]).read_bytes()[:80]
            matches = [env for env in matches if env["code"][:80] == header]
        if matches:
            environment = max(matches, key=lambda env: env["version"])
            env_path, key_path = records / f"{identity}.env", records / f"{identity}.key"
            env_path.write_bytes(environment["bytes"])
            key_path.write_bytes(environment["key"])
            command += ["--environment", str(env_path), "--cache-version", str(environment["version"])]
            if base["stage"] != "compute":
                command += ["--pipeline-key", str(key_path)]
            base["context_source"] = "inferred-compatible-cache-environment"
            base["cache_candidates"] = [{key: env[key] for key in ("source", "entry", "index", "offset", "version")}
                                        for env in matches]
            base["selected_cache_context"] = {key: environment[key] for key in
                                              ("source", "entry", "index", "offset", "version")}
            base["cache_version"] = environment["version"]
        elif base["stage"] == "compute" and args.static_context:
            command += ["--control", base["control"], "--allocation", base["source"]]
            base["context_source"] = "static-control-launch-with-static-driver-context"
        elif base["stage"] == "compute":
            return {**base, "status": "partial", "phase": "context", "lowered": True,
                    "error": {"message": "compute launch and resource context not established"}}
        elif args.static_context:
            command += ["--control", base["control"]]
            base["context_source"] = "static-header-with-static-driver-context"
        else:
            base["context_source"] = "static-header-with-synthetic-pipeline-defaults"
        output = records / f"{identity}.spv"
        result = invoke(command + ["--output", str(output)], args.timeout)
        if result["status"] == "ok":
            try:
                validated = subprocess.run([validator, "--target-env", "vulkan1.2", str(output)],
                                           capture_output=True, text=True, timeout=args.timeout)
                if validated.returncode:
                    result.update(status="error", phase="validate", error={"message": validated.stderr.strip()})
                elif matches:
                    result.update(status="partial", phase="validated-compatible-cache")
                else:
                    result.update(status="partial", phase="validated-synthetic")
            except subprocess.TimeoutExpired:
                result.update(status="error", phase="timeout", error={"message": "spirv-val timeout"})
        return {**base, **result, "lowered": True, "command": command + ["--output", str(output)]}

    results = []
    with (args.output / "results.jsonl").open("w", encoding="utf-8") as journal:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            for index, result in enumerate(executor.map(compile_one, programs.values()), 1):
                results.append(result)
                journal.write(json.dumps(result) + "\n")
                journal.flush()
                if index % 250 == 0:
                    print(f"Processed {index}/{len(programs)}", file=sys.stderr, flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (args.output / "archive-errors.json").write_text(json.dumps(archive_errors + cache_errors, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.md").write_text(summarize(results), encoding="utf-8")
    print(json.dumps({"programs": len(results), "lowered": sum(row["lowered"] for row in results),
                      "phases": dict(Counter(row["phase"] for row in results)),
                      "archive_errors": len(archive_errors), "cache_errors": len(cache_errors)}))
    return int(any(row["status"] != "ok" for row in results) or bool(archive_errors or cache_errors))


if __name__ == "__main__":
    sys.exit(main())
