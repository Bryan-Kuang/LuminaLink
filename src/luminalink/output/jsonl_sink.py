from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from luminalink.types import NarrationEvent


class JSONLSink:
    """Write narration events as JSON Lines."""

    def __init__(self, path: Optional[str]):
        self._path = path
        self._fp = None
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fp = p.open("w", encoding="utf-8")

    def emit(self, event: NarrationEvent) -> None:
        """Write one event to file."""

        if self._fp is None:
            return
        self._fp.write(json.dumps(event.model_dump(), ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        """Close the underlying file handle."""

        if self._fp is not None:
            self._fp.close()
            self._fp = None

