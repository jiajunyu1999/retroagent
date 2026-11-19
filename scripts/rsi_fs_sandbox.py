"""Runtime support for the frozen V9 controller."""

import ctypes

import os

from pathlib import Path

HANDLED = (1 << 13) - 1  # Landlock ABI1, supported by the current5.15 kernel.

def restrict(read_paths, write_paths, uid=65534):
    libc = ctypes.CDLL(None, use_errno=True)

    def checked(value, operation):
        if value < 0:
            raise OSError(ctypes.get_errno(), operation)
        return value

    class Ruleset(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]

    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]

    attr = Ruleset(HANDLED)
    ruleset = checked(libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0), 'Landlock ruleset')
    paths = {}
    for path in read_paths:
        path = Path(path).resolve()
        if path.exists():
            paths[path] = paths.get(path, 0) | READ_FILE | EXECUTE | (READ_DIR if path.is_dir() else 0)
    for path in write_paths:
        path = Path(path).resolve()
        if path.exists():
            paths[path] = HANDLED if path.is_dir() else READ_FILE | 2
    for path, access in paths.items():
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            rule = Beneath(access, fd)
            checked(libc.syscall(445, ruleset, 1, ctypes.byref(rule), 0), 'Landlock allow rule')
        finally:
            os.close(fd)
    checked(libc.prctl(38, 1, 0, 0, 0), 'no_new_privs')
    # Remove capabilities and change identity before executing any model process.
    for capability in range(64):
        libc.prctl(24, capability, 0, 0, 0)
    os.setgroups([])
    os.setresgid(uid, uid, uid)
    os.setresuid(uid, uid, uid)
    checked(libc.syscall(446, ruleset, 0), 'Landlock restrict_self')
    os.close(ruleset)

def system_reads():
    return [Path(p) for p in [
        '/usr', '/bin', '/lib', '/lib64', '/opt/conda',
        '/etc/ssl/certs', '/etc/ssl/openssl.cnf', '/etc/resolv.conf', '/etc/hosts', '/etc/nsswitch.conf',
        '/etc/passwd', '/etc/group', '/etc/ld.so.cache', '/etc/localtime',
        '/etc/profile', '/etc/profile.d', '/dev/null', '/dev/urandom', '/dev/random',
        '/proc/meminfo', '/proc/cpuinfo', '/proc/version',
        '/proc/sys/kernel/random/boot_id']]
