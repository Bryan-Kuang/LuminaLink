"""
Camera GUI Application

Main graphical user interface for real-time camera narration.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import logging
import time
from pathlib import Path
from typing import Optional

from ..camera_controller import CameraRealtimeController

logger = logging.getLogger(__name__)


class CameraApp:
    """
    Main GUI application for camera narration.

    Provides video preview, controls, status display, and settings.
    """

    def __init__(
        self,
        camera_index: int = 0,
        characters_config: Optional[str] = None,
        mic_device_index: int = 0,
        cooldown: float = 5.0,
    ):
        """
        Initialize camera application.

        Args:
            camera_index: Camera device index
            characters_config: Path to character configuration JSON
            mic_device_index: Microphone device index
            cooldown: Minimum seconds between narrations
        """
        self.camera_index = camera_index
        self.characters_config = characters_config

        # Controller
        self.controller = CameraRealtimeController(
            camera_index=camera_index,
            characters_config=characters_config,
            mic_device_index=mic_device_index,
            cooldown=cooldown,
        )

        # Tkinter setup
        self.root = tk.Tk()
        self.root.title("LuminaLink Camera Narration")
        self.root.geometry("800x750")
        self.root.resizable(True, True)

        # Current frame reference
        self.current_photo = None

        # UI components
        self._setup_ui()
        self._bind_callbacks()
        self._bind_shortcuts()

        # Protocol for window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _setup_ui(self):
        """Build UI components"""
        # Video preview canvas
        preview_frame = ttk.Frame(self.root)
        preview_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.video_canvas = tk.Canvas(
            preview_frame,
            width=640,
            height=480,
            bg='black',
            highlightthickness=0
        )
        self.video_canvas.pack()

        # Camera info label
        self.camera_info_label = ttk.Label(
            self.root,
            text=f"Camera {self.camera_index}",
            font=('Arial', 10)
        )
        self.camera_info_label.pack(pady=2)

        # Control buttons frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)

        self.start_btn = ttk.Button(
            control_frame,
            text="▶ Start",
            command=self.start,
            width=12
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.pause_btn = ttk.Button(
            control_frame,
            text="⏸ Pause",
            command=self.pause,
            state=tk.DISABLED,
            width=12
        )
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            control_frame,
            text="⏹ Stop",
            command=self.stop,
            state=tk.DISABLED,
            width=12
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Status frame
        status_frame = ttk.LabelFrame(self.root, text="Status", padding=5)
        status_frame.pack(pady=5, padx=10, fill=tk.X)

        self.status_label = ttk.Label(
            status_frame,
            text="Status: Ready",
            font=('Arial', 11, 'bold'),
            foreground='green'
        )
        self.status_label.pack()

        # Statistics frame
        stats_frame = ttk.Frame(status_frame)
        stats_frame.pack(pady=5)

        self.stats_label = ttk.Label(
            stats_frame,
            text="Frames: 0 | Narrations: 0",
            font=('Arial', 9)
        )
        self.stats_label.pack()

        # Narration log frame
        log_frame = ttk.LabelFrame(self.root, text="Narration Log", padding=5)
        log_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(
            log_frame,
            height=8,
            width=80,
            state=tk.DISABLED,
            font=('Arial', 10),
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)

        # Help text
        help_text = ttk.Label(
            self.root,
            text="Shortcuts: Space = Pause/Resume | Esc = Stop",
            font=('Arial', 9),
            foreground='gray'
        )
        help_text.pack(pady=5)

    def _bind_callbacks(self):
        """Connect controller callbacks to GUI"""
        self.controller.set_on_frame_callback(self.update_video_frame)
        self.controller.set_on_narration_callback(self.display_narration)
        self.controller.set_on_status_callback(self.update_status)

    def _bind_shortcuts(self):
        """Keyboard shortcuts"""
        self.root.bind("<space>", lambda e: self.pause())
        self.root.bind("<Escape>", lambda e: self.stop())
        self.root.bind("<Control-q>", lambda e: self._on_closing())

    # Event handlers
    def start(self):
        """Start narration"""
        try:
            self.controller.start()

            # Update UI
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)

            # Start statistics update loop
            self._update_statistics()

            logger.info("Application started")
        except Exception as e:
            logger.error(f"Failed to start: {e}")
            messagebox.showerror(
                "Start Error",
                f"Failed to start camera narration:\n\n{str(e)}\n\n"
                "Please check:\n"
                "1. Camera is connected and not in use\n"
                "2. Microphone permissions are granted\n"
                "3. OpenAI API key is configured"
            )

    def pause(self):
        """Toggle pause"""
        if not self.controller.state.is_playing:
            return

        if self.controller.state.is_paused:
            self.controller.resume()
            self.pause_btn.config(text="⏸ Pause")
            logger.info("Application resumed")
        else:
            self.controller.pause()
            self.pause_btn.config(text="▶ Resume")
            logger.info("Application paused")

    def stop(self):
        """Stop narration"""
        self.controller.stop()

        # Update UI
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="⏸ Pause")
        self.stop_btn.config(state=tk.DISABLED)

        # Clear canvas
        self.video_canvas.delete("all")
        self.video_canvas.create_text(
            320, 240,
            text="Camera Stopped",
            fill='white',
            font=('Arial', 20)
        )

        logger.info("Application stopped")

    def update_video_frame(self, frame_bgr: np.ndarray):
        """
        Update video preview (called from camera thread).

        Args:
            frame_bgr: BGR image from camera
        """
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Resize to canvas size
            frame_resized = cv2.resize(frame_rgb, (640, 480))

            # Convert to PIL Image
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image=image)

            # Schedule UI update in main thread
            self.root.after(0, self._update_canvas, photo)

        except Exception as e:
            logger.error(f"Error updating frame: {e}")

    def _update_canvas(self, photo):
        """Actually update canvas (main thread only)"""
        try:
            self.video_canvas.delete("all")
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.current_photo = photo  # Keep reference to prevent garbage collection
        except Exception as e:
            logger.error(f"Error updating canvas: {e}")

    def display_narration(self, text: str):
        """
        Display narration text (subtitle + log).

        Args:
            text: Narration text
        """
        # Add subtitle overlay
        self.root.after(0, self._add_subtitle, text)

        # Add to log
        self.root.after(0, self._add_to_log, text)

    def _add_subtitle(self, text: str):
        """Draw subtitle on canvas"""
        try:
            # Draw semi-transparent background
            self.video_canvas.create_rectangle(
                0, 420, 640, 480,
                fill='black',
                stipple='gray50',
                tags='subtitle'
            )

            # Draw text
            self.video_canvas.create_text(
                320, 450,
                text=text,
                fill='white',
                font=('Arial', 12, 'bold'),
                width=600,
                anchor=tk.CENTER,
                tags='subtitle'
            )

            # Auto-clear after 4 seconds
            self.root.after(4000, self._clear_subtitle)

        except Exception as e:
            logger.error(f"Error adding subtitle: {e}")

    def _clear_subtitle(self):
        """Clear subtitle overlay"""
        try:
            self.video_canvas.delete('subtitle')
        except Exception as e:
            logger.error(f"Error clearing subtitle: {e}")

    def _add_to_log(self, text: str):
        """Add narration to log"""
        try:
            timestamp = time.strftime("%H:%M:%S")

            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"[{timestamp}] {text}\n")
            self.log_text.see(tk.END)  # Auto-scroll
            self.log_text.config(state=tk.DISABLED)

        except Exception as e:
            logger.error(f"Error adding to log: {e}")

    def update_status(self, status: str):
        """
        Update status label.

        Args:
            status: Status text (Running/Paused/Stopped)
        """
        self.root.after(0, self._update_status_label, status)

    def _update_status_label(self, status: str):
        """Actually update status label (main thread only)"""
        try:
            # Color coding
            color = 'green' if status == 'Running' else \
                    'orange' if status == 'Paused' else 'red'

            self.status_label.config(
                text=f"Status: {status}",
                foreground=color
            )
        except Exception as e:
            logger.error(f"Error updating status: {e}")

    def _update_statistics(self):
        """Update statistics display"""
        if not self.controller.state.is_playing:
            return

        try:
            stats = self.controller.get_statistics()
            self.stats_label.config(
                text=f"Frames: {stats['frame_count']} | "
                     f"Narrations: {stats['narration_count']}"
            )

            # Schedule next update
            self.root.after(1000, self._update_statistics)

        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

    def _on_closing(self):
        """Handle window close event"""
        if self.controller.state.is_playing:
            if messagebox.askokcancel(
                "Quit",
                "Narration is still running. Are you sure you want to quit?"
            ):
                self.stop()
                self.root.destroy()
        else:
            self.root.destroy()

    def run(self):
        """Start GUI event loop"""
        logger.info("Starting LuminaLink Camera App")
        self.root.mainloop()
        logger.info("LuminaLink Camera App closed")
