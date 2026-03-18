"""
Audio Input Module

Captures microphone audio in real-time and feeds it to the audio detector.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    logger.warning("sounddevice not available. Audio input will be disabled.")


class AudioInputStream:
    """
    Captures microphone audio and feeds to RealtimeAudioDetector.

    This class uses sounddevice to capture audio from the system microphone
    in real-time and feeds it to a RealtimeAudioDetector for silence detection.
    """

    def __init__(self, detector, sample_rate: int = 22050, blocksize: int = 512, device_id: Optional[int] = None):
        """
        Initialize audio input stream.

        Args:
            detector: RealtimeAudioDetector instance to feed audio to
            sample_rate: Audio sample rate in Hz (default: 22050)
            blocksize: Audio block size in frames (default: 512, ~23ms at 22050Hz)
            device_id: Audio input device ID (None for system default)
        """
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError(
                "sounddevice library not available. "
                "Install it with: pip install sounddevice"
            )

        self.detector = detector
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.device_id = device_id
        self.stream: Optional[sd.InputStream] = None
        self._running = False

    def _audio_callback(self, indata, frames, time, status):
        """
        Audio callback function called by sounddevice.

        Args:
            indata: Input audio data (numpy array of shape (frames, channels))
            frames: Number of frames
            time: Time information
            status: Stream status
        """
        if status:
            logger.warning(f"Audio input status: {status}")

        # Convert to mono if stereo
        if indata.ndim > 1:
            audio_mono = indata[:, 0]  # Take first channel
        else:
            audio_mono = indata

        # Feed to detector
        try:
            self.detector.feed_audio(audio_mono)
        except Exception as e:
            logger.error(f"Error feeding audio to detector: {e}")

    def start(self):
        """Start audio capture."""
        if self._running:
            logger.warning("Audio stream already running")
            return

        try:
            self.stream = sd.InputStream(
                device=self.device_id,
                samplerate=self.sample_rate,
                channels=1,  # Mono audio
                callback=self._audio_callback,
                blocksize=self.blocksize,
            )
            self.stream.start()
            self._running = True
            logger.info(
                f"Audio stream started: {self.sample_rate}Hz, "
                f"blocksize={self.blocksize}"
            )
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            raise

    def stop(self):
        """Stop audio capture."""
        if not self._running:
            return

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                logger.info("Audio stream stopped")
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
            finally:
                self.stream = None
                self._running = False

    def is_running(self) -> bool:
        """Check if audio stream is running."""
        return self._running

    def get_device_info(self, device_id: Optional[int] = None) -> dict:
        """
        Get information about audio input device.

        Args:
            device_id: Device ID (None for default device)

        Returns:
            Device information dictionary
        """
        if not SOUNDDEVICE_AVAILABLE:
            return {}

        try:
            if device_id is None:
                device_id = sd.default.device[0]  # Default input device
            return sd.query_devices(device_id)
        except Exception as e:
            logger.error(f"Error querying device info: {e}")
            return {}

    @staticmethod
    def list_devices():
        """
        List all available audio input devices.

        Returns:
            List of device information dictionaries
        """
        if not SOUNDDEVICE_AVAILABLE:
            logger.error("sounddevice not available")
            return []

        try:
            devices = sd.query_devices()
            input_devices = [
                {
                    "id": i,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "sample_rate": dev["default_samplerate"],
                }
                for i, dev in enumerate(devices)
                if dev["max_input_channels"] > 0
            ]
            return input_devices
        except Exception as e:
            logger.error(f"Error listing devices: {e}")
            return []

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
