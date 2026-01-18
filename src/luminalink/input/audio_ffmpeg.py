from __future__ import annotations

import subprocess
from collections.abc import Iterator

from luminalink.types import AudioFrame


class FFmpegAudioInput:
    """Stream PCM audio frames from a video file via ffmpeg."""

    def __init__(
        self,
        video_path: str,
        sample_rate_hz: int = 16000,
        frame_ms: int = 30,
    ):
        self._video_path = video_path
        self._sample_rate_hz = sample_rate_hz
        self._frame_ms = frame_ms
        self._proc: subprocess.Popen[bytes] | None = None

        if frame_ms not in (10, 20, 30):
            raise ValueError("WebRTC VAD expects 10/20/30ms frames")

    def frames(self) -> Iterator[AudioFrame]:
        """Yield fixed-duration PCM frames as `AudioFrame` objects."""

        bytes_per_sample = 2
        samples_per_frame = int(self._sample_rate_hz * (self._frame_ms / 1000.0))
        bytes_per_frame = samples_per_frame * bytes_per_sample

        cmd = [
            "ffmpeg",
            "-i",
            self._video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self._sample_rate_hz),
            "-f",
            "s16le",
            "-loglevel",
            "error",
            "pipe:1",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert self._proc.stdout is not None

        frame_index = 0
        while True:
            buf = self._proc.stdout.read(bytes_per_frame)
            if not buf or len(buf) < bytes_per_frame:
                break
            start_ms = int(frame_index * self._frame_ms)
            end_ms = int((frame_index + 1) * self._frame_ms)
            yield AudioFrame(
                start_ms=start_ms,
                end_ms=end_ms,
                pcm_s16le=buf,
                sample_rate_hz=self._sample_rate_hz,
            )
            frame_index += 1

        self.close()

    def close(self) -> None:
        """Terminate the underlying ffmpeg process."""

        if self._proc is None:
            return
        try:
            self._proc.kill()
        finally:
            self._proc = None

