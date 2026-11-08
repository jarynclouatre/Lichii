#!/usr/bin/env python3
"""Build a compute warm-up file from a static shader dump.

For every compute program in the dump manifest (extract_static_shaders.py output) this
lowers the native code with shader-sweep (--capture-extent, so the program is cut where the
emulator cuts it at runtime), rebuilds the exact code layout the emulator stores in its
pipeline cache, and writes one cache-format entry per program:

    'yuzucach' + u32 version, then per entry:
    u32 env count (1), env record with zero recorded reads, compute trailer, 24-byte key.

Env record fields match what the emulator writes for native compute shaders:
    local_memory_size = control u32 at byte 104 (QMD local alloc)
    texture_bound     = 0 (tex_cb_index, 0 in every runtime record seen)
    start_address     = 80 (sizeof the 10-word header, compute header is all zero)
    read_lowest       = 0, read_highest = code size - 8
    viewport_transform_state = 1, stage = 5
Compute trailer: workgroup x,y,z (control bytes 88,92,96) and shared memory (control byte
100 rounded up to 256). Key: CityHash64(code) ^ 0x4f554e4345000002, shared, workgroup.

Needs the `cityhash` package (pip install cityhash).
"""
import argparse
import json
import os
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cityhash import CityHash64

MAGIC = 0x50414d4e4544454E
HASH_XOR = 0x4F554E4345000002
ENV = struct.Struct('<6Q7I')
COMPUTE_STAGE = 5
HEADER_WORDS = 10
BELOW_NORMAL = 0x4000 if os.name == 'nt' else 0


def runtime_code(instructions, offsets):
    """Same layout as LowerNativeCode (video_core/shader_environment.cpp) for compute."""
    code = [0] * HEADER_WORDS
    for instruction in instructions:
        if len(code) % 4 == 0:
            code.append(0)  # Maxwell scheduling word
        code.append(instruction)
    code.append(0)
    if offsets:
        # AppendNativeAddressMap: lowered index -> byte address inside `code`.
        code.extend(0x50 + i * 8 + ((i + 1) // 3) * 8 for i in offsets)
        code.append(len(offsets))
        code.append(MAGIC)
    return struct.pack(f'<{len(code)}Q', *code)


def read_override(path):
    """Instructions and address map offsets of a lowered override record; raises on others."""
    data = Path(path).read_bytes()
    magic, version, source_size, count, _texture_buffer, map_count = struct.unpack_from('<6I', data)
    if magic != 0x4E454445 or version != 0x3230564F or map_count != source_size // 16:
        raise ValueError(f'{path}: not a mapped override record')
    at = 24 + source_size
    instructions = struct.unpack_from(f'<{count}Q', data, at)
    offsets = struct.unpack_from(f'<{map_count}I', data, at + count * 8)
    return instructions, offsets


def control_values(control):
    workgroup = struct.unpack_from('<3I', control, 88)
    shared = (struct.unpack_from('<I', control, 100)[0] + 255) // 256 * 256
    local = struct.unpack_from('<I', control, 104)[0]
    return workgroup, shared, local


def entry(code, workgroup, shared, local):
    env = ENV.pack(len(code), 0, 0, 0, 0, 0, local, 0, HEADER_WORDS * 8, 0, len(code) - 8, 1,
                   COMPUTE_STAGE)
    trailer = struct.pack('<4I', *workgroup, shared)
    key = struct.pack('<QI3I', CityHash64(code) ^ HASH_XOR, shared, *workgroup)
    return struct.pack('<I', 1) + env + code + trailer + key, key


def main():
    """Lowers the dump's compute programs with shader-sweep and writes one entry per unique key."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--corpus', default='./corpus',
                    help='static shader dump folder (extract_static_shaders.py output)')
    ap.add_argument('--sweep', default='shader-sweep', help='shader-sweep executable')
    ap.add_argument('--work', default='./work/compute-ovr',
                    help='folder for the lowered override records (reused between runs)')
    ap.add_argument('--out', default='./warm-compute.bin')
    ap.add_argument('--version', type=int, default=54)
    ap.add_argument('--jobs', type=int, default=6)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    programs = [p for p in json.loads((corpus / 'manifest.json').read_text())['programs']
                if p['stage'] == 'compute']

    def lower(program):
        """Lowers one program (reusing its record); returns it with the record, None on failure."""
        record = work / f"{program['code_sha256']}.ovr"
        if not record.exists():
            # A hung or unstartable sweep counts as a failed program, as in make_graphics_warm.py,
            # instead of stopping the whole run.
            try:
                subprocess.run([args.sweep, str(corpus / program['code']), '--capture-extent',
                                '--lower-only', str(work / f"{program['code_sha256']}.words"),
                                '--override-out', str(record)],
                               capture_output=True, timeout=300, creationflags=BELOW_NORMAL)
            except (subprocess.TimeoutExpired, OSError) as ex:
                print(f"{program['code_sha256']}: {type(ex).__name__}", file=sys.stderr)
        return program, record if record.exists() else None

    with ThreadPoolExecutor(args.jobs) as pool:
        lowered = list(pool.map(lower, programs))

    entries = {}
    failed = 0
    for program, record in lowered:
        if record is None:
            failed += 1
            continue
        code = runtime_code(*read_override(record))
        workgroup, shared, local = control_values((corpus / program['control']).read_bytes())
        if not all(workgroup):
            failed += 1
            continue
        data, key = entry(code, workgroup, shared, local)
        entries.setdefault(key, data)

    out = Path(args.out)
    with open(out, 'wb') as f:
        f.write(b'yuzucach' + struct.pack('<I', args.version))
        for data in entries.values():
            f.write(data)
    print(json.dumps({'programs': len(programs), 'failed': failed, 'entries': len(entries),
                      'output': str(out), 'bytes': out.stat().st_size}))


if __name__ == '__main__':
    sys.exit(main())
