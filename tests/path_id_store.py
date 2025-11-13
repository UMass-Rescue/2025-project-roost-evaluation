import json
import os
from pathlib import Path
from typing import Iterable


def _canonicalize_path(path_str: str) -> str:
    return Path(path_str).resolve().as_posix()


class PathIdStore:
    """
    Maintains a mapping from canonical file paths to stable numeric IDs.
    - If mapping_file is provided, loads on init (if exists) and saves on demand.
    - If mapping_file is None, operates purely in-memory.
    """

    def __init__(self, mapping_file: str | None = None):
        self._file: Path | None = Path(mapping_file) if mapping_file else None
        self._map: dict[str, str] = {}
        self._next_id_value: int = 1
        if self._file is not None and self._file.exists():
            self._load()
        self._recompute_next_id()

    @classmethod
    def from_env(cls) -> "PathIdStore":
        path = os.getenv("ANON_ID_MAP_FILEPATH")
        return cls(path.strip() if path else None)

    @property
    def has_persistence(self) -> bool:
        return self._file is not None

    def _load(self) -> None:
        try:
            with open(self._file, "r") as f:  # type: ignore[arg-type]
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(k, str) and isinstance(v, str):
                        canon = _canonicalize_path(k)
                        self._map[canon] = v
        except Exception:
            # If load fails, start with empty map
            self._map = {}

    def _recompute_next_id(self) -> None:
        max_id = 0
        for v in self._map.values():
            try:
                n = int(v)
            except Exception:
                continue
            if n > max_id:
                max_id = n
        self._next_id_value = max_id + 1

    def _next_id(self) -> str:
        value = str(self._next_id_value)
        self._next_id_value += 1
        return value

    def get_or_assign_id_for_path(self, path_str: str) -> str:
        canon = _canonicalize_path(path_str)
        if canon in self._map:
            return self._map[canon]
        new_id = self._next_id()
        self._map[canon] = new_id
        return new_id

    def get_ids_for_paths(self, paths: Iterable[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for p in paths:
            result[p] = self.get_or_assign_id_for_path(p)
        return result

    def save(self) -> None:
        if self._file is None:
            return
        self._file.parent.mkdir(parents=True, exist_ok=True)
        # Writing to a temporary file first to avoid corruption if the process is interrupted
        tmp_path = self._file.with_suffix(self._file.suffix + ".tmp")
        data = {k: v for k, v in self._map.items()}
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self._file)


