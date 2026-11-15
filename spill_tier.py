"""Optional third tier: exact copies of routed experts on a faster local drive (v0.7).

Why: a RAM cache can only remove *repeated* reads. An expert the conversation has
not touched yet has to come from storage, and on a USB link that is the slow part.
If the machine has an internal SSD with spare room, experts that were read once
from the model drive are also written there, and later reads (this session or a
later one) come from the fast drive instead. Nothing is approximated: a spill
file holds the checkpoint's bytes for one expert and is only used when

  - it names the same shard files, offsets and sizes as the catalog,
  - those shard files still have the size and modification time recorded, and
  - the payload matches its stored CRC-32.

Anything else (missing, truncated, stale, corrupt) is ignored and the expert is
read from the checkpoint as before. The checkpoint itself is never written.

Off unless --spill-dir is given. The tier fills up to its byte budget and then
stops admitting; it never deletes. Remove the folder to reset it.

STATUS: unit-tested on Linux in the development sandbox. UNRUN on Windows and
UNMEASURED: whether it helps depends on the spill drive really being faster than
the model drive. bench_e2e.py --spill-dir measures that.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
import threading
import zlib

MAGIC = b'FRSPILL1'
HEADER_ALLOWANCE = 1024  # reserved per file at admission, corrected to the real size once written


class SpillTier:
    def __init__(self, root, budget_bytes, checkpoint_id, verify=True, max_queued=8):
        self.root = Path(root) / hashlib.sha256(str(checkpoint_id).encode()).hexdigest()[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        self.budget = int(budget_bytes)
        self.verify = verify
        self.max_queued = max_queued
        self._lock = threading.Lock()
        self._writer = ThreadPoolExecutor(1, thread_name_prefix='frustum-spill')
        self._queued = 0
        self._stat = {}
        self.used = sum(entry.stat().st_size for entry in self.root.rglob('*.bin'))
        self.hits = self.hit_bytes = self.stores = self.rejected = self.skipped_full = self.skipped_busy = 0
        self.write_errors = 0

    def _path(self, key):
        return self.root / str(key[0]) / f'{key[1]}.bin'

    def identity(self, items):
        """What must be unchanged for a spill file to stand in for these tensors."""
        out = []
        for item in items:
            stat = self._stat.get(item.path)
            if stat is None:
                info = os.stat(item.path)
                stat = self._stat[item.path] = (info.st_size, info.st_mtime_ns)
            out.append([item.path.name, item.offset, item.size, stat[0], stat[1]])
        return out

    def load(self, key, identity):
        """The expert's buffers in catalog order, or None when no valid copy exists."""
        path = self._path(key)
        try:
            data = bytearray(path.stat().st_size)
            with open(path, 'rb', buffering=0) as handle:
                view, done = memoryview(data), 0
                while done < len(data):
                    got = handle.readinto(view[done:])
                    if not got:
                        break
                    done += got
            if done != len(data) or data[:8] != MAGIC:
                raise ValueError('short or foreign file')
            length, = struct.unpack_from('<I', data, 8)
            header = json.loads(bytes(data[12:12 + length]))
            payload = view[12 + length:]
            if header['identity'] != identity or len(payload) != sum(row[2] for row in identity):
                raise ValueError('stale')
            if self.verify and zlib.crc32(payload) != header['crc32']:
                raise ValueError('checksum')
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, struct.error):
            with self._lock:
                self.rejected += 1
            return None
        buffers, start = [], 0
        for row in identity:
            buffers.append(payload[start:start + row[2]])
            start += row[2]
        with self._lock:
            self.hits += 1
            self.hit_bytes += len(payload)
        return buffers

    def store(self, key, buffers, identity):
        """Queue an exact copy for writing. Never blocks the reader; drops the copy when busy or full."""
        size = sum(len(buffer) for buffer in buffers) + HEADER_ALLOWANCE
        with self._lock:
            if self.used + size > self.budget:
                self.skipped_full += 1
                return False
            if self._queued >= self.max_queued:
                self.skipped_busy += 1
                return False
            self._queued += 1
            self.used += size
        self._writer.submit(self._write, key, buffers, identity, size)
        return True

    def _write(self, key, buffers, identity, size):
        path = self._path(key)
        temporary = path.with_suffix(f'.tmp{threading.get_ident()}')
        try:
            crc = 0
            for buffer in buffers:
                crc = zlib.crc32(buffer, crc)
            header = json.dumps(dict(identity=identity, crc32=crc)).encode()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(temporary, 'wb') as handle:
                handle.write(MAGIC + struct.pack('<I', len(header)) + header)
                for buffer in buffers:
                    handle.write(buffer)
            os.replace(temporary, path)  # readers see the old file or the complete new one
            with self._lock:
                self.stores += 1
                self.used += 12 + len(header) - HEADER_ALLOWANCE  # the budget counts whole files
        except OSError:
            with self._lock:
                self.write_errors += 1
                self.used -= size
        finally:
            with self._lock:
                self._queued -= 1

    def stats(self):
        with self._lock:
            return dict(directory=str(self.root), budget_bytes=self.budget, used_bytes=self.used, hits=self.hits,
                        hit_bytes=self.hit_bytes, stores=self.stores, rejected=self.rejected,
                        skipped_full=self.skipped_full, skipped_busy=self.skipped_busy,
                        write_errors=self.write_errors, verify=self.verify)

    def close(self):
        self._writer.shutdown(wait=True)
