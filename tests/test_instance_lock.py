import pytest

from aitrader.core.instance_lock import (
    InstanceAlreadyRunningError,
    InstanceLock,
)


def test_instance_lock_excludes_second_runtime_and_releases(tmp_path):
    lock_path = tmp_path / "data" / "runtime.lock"
    first = InstanceLock(lock_path)
    second = InstanceLock(lock_path)

    first.acquire()
    try:
        with pytest.raises(InstanceAlreadyRunningError):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()


def test_instance_lock_release_is_safe_when_unacquired(tmp_path):
    lock = InstanceLock(tmp_path / "runtime.lock")

    lock.release()

    with lock:
        assert lock.path.exists()
