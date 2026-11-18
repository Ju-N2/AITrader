import fcntl
from pathlib import Path
from types import TracebackType


class InstanceAlreadyRunningError(RuntimeError):
    """Raised when another process already owns the runtime lock."""


class InstanceLock:
    """Prevent two local AITrader runtimes from sharing one trading DB."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = None

    def acquire(self) -> None:
        if self._file is not None:
            raise RuntimeError("Instance lock is already acquired")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.path.open("a", encoding="utf-8")

        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock_file.close()
            raise InstanceAlreadyRunningError(
                "Another AITrader runtime is already running"
            ) from error
        except BaseException:
            lock_file.close()
            raise

        self._file = lock_file

    def release(self) -> None:
        lock_file = self._file
        if lock_file is None:
            return

        self._file = None
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()
