"""OS-level helpers for honest storage benchmarks. No torch, no model code.

DeviceCounters   cumulative bytes the *device* delivered, from the OS:
                 Windows  IOCTL_DISK_PERFORMANCE on the volume (needs no admin rights;
                          returns None if disk performance counters are disabled)
                 Linux    /sys/dev/block/<major>:<minor>/stat
                 These are device-wide: other programs reading the same drive are
                 included. Close them during a benchmark.
purge_file_cache best effort removal of the given files from the OS file cache so the
                 next read is cold.
                 Windows  opening a file with FILE_FLAG_NO_BUFFERING and closing it makes
                          the cache manager drop that file's cached pages. This is observed
                          behaviour, not a documented contract, so bench_e2e.py verifies
                          every cold run by comparing physical with logical bytes instead
                          of trusting the purge.
                 Linux    posix_fadvise(DONTNEED).

STATUS: the Linux paths ran in the development sandbox. The Windows paths are UNRUN.
"""
import ctypes
import os
from pathlib import Path


def _windows_volume(path):
    drive = os.path.splitdrive(str(Path(path).resolve()))[0]
    if len(drive) != 2 or drive[1] != ':':
        raise OSError('Only drive-letter volumes are supported: ' + str(path))
    return '\\\\.\\' + drive


class DeviceCounters:
    """read() -> dict(bytes_read, read_count, read_time_seconds) or None when unavailable."""
    def __init__(self, path):
        self.path = Path(path)
        self.error = None
        self._stat = None
        if os.name != 'nt':
            try:
                dev = os.stat(self.path).st_dev
                base = Path(f'/sys/dev/block/{os.major(dev)}:{os.minor(dev)}')
                # A partition's counters live in its own stat file; both forms work.
                self._stat = base / 'stat' if (base / 'stat').exists() else None
                if self._stat is None:
                    self.error = 'no block-device statistics for this filesystem'
            except OSError as error:
                self.error = str(error)

    def read(self):
        try:
            if os.name == 'nt':
                return self._read_windows()
            if self._stat is None:
                return None
            fields = self._stat.read_text().split()
            return dict(bytes_read=int(fields[2]) * 512, read_count=int(fields[0]),
                        read_time_seconds=int(fields[3]) / 1000)
        except (OSError, ValueError, IndexError) as error:
            self.error = str(error)
            return None

    def _read_windows(self):
        from ctypes import wintypes

        class DiskPerformance(ctypes.Structure):
            _fields_ = [('BytesRead', ctypes.c_longlong), ('BytesWritten', ctypes.c_longlong),
                        ('ReadTime', ctypes.c_longlong), ('WriteTime', ctypes.c_longlong),
                        ('IdleTime', ctypes.c_longlong), ('ReadCount', wintypes.DWORD),
                        ('WriteCount', wintypes.DWORD), ('QueueDepth', wintypes.DWORD),
                        ('SplitCount', wintypes.DWORD), ('QueryTime', ctypes.c_longlong),
                        ('StorageDeviceNumber', wintypes.DWORD),
                        ('StorageManagerName', ctypes.c_wchar * 8)]

        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                           ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                                           ctypes.c_void_p]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        # Zero desired access is enough for this query and needs no elevation.
        handle = kernel.CreateFileW(_windows_volume(self.path), 0, 0x1 | 0x2, None, 3, 0, None)
        if handle == wintypes.HANDLE(-1).value:
            raise OSError(ctypes.get_last_error(), 'Cannot open volume for performance query')
        try:
            data, returned = DiskPerformance(), wintypes.DWORD()
            if not kernel.DeviceIoControl(handle, 0x00070020, None, 0, ctypes.byref(data),
                                          ctypes.sizeof(data), ctypes.byref(returned), None):
                raise OSError(ctypes.get_last_error(), 'IOCTL_DISK_PERFORMANCE failed (try: diskperf -y)')
            return dict(bytes_read=int(data.BytesRead), read_count=int(data.ReadCount),
                        read_time_seconds=data.ReadTime / 1e7)
        finally:
            kernel.CloseHandle(handle)


def delta(before, after):
    """Difference of two DeviceCounters.read() results, or None if either is missing."""
    if before is None or after is None:
        return None
    return {key: after[key] - before[key] for key in after}


def purge_file_cache(paths):
    """Ask the OS to forget cached pages of these files. Returns (purged, failed)."""
    purged = failed = 0
    for path in paths:
        try:
            if os.name == 'nt':
                from ctypes import wintypes
                kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                kernel.CreateFileW.restype = wintypes.HANDLE
                kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                               wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
                kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                # GENERIC_READ, share read/write/delete, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING
                handle = kernel.CreateFileW(str(path), 0x80000000, 0x7, None, 3, 0x20000000, None)
                if handle == wintypes.HANDLE(-1).value:
                    raise OSError(ctypes.get_last_error(), 'open failed')
                kernel.CloseHandle(handle)
            else:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            purged += 1
        except OSError:
            failed += 1
    return purged, failed
