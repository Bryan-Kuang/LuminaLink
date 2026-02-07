"""
LuminaLink - AI Movie Audio Description for the Visually Impaired

Main program entry point
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, List
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from .config import get_config, reload_config
from .video_processor import VideoProcessor, VideoFrame
from .audio_detector import AudioDetector
from .character_recognizer import CharacterRecognizer, CharacterTracker
from .scene_analyzer import SceneAnalyzer
from .narrator import Narrator, NarrationStyle
from .tts_engine import TTSManager, NarrationPlayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

console = Console()


class LuminaLink:
    """LuminaLink Main Application Class"""
    
    def __init__(
        self,
        video_path: str,
        characters_config: Optional[str] = None,
        preview: bool = False,
        output_subtitles: Optional[str] = None
    ):
        """
        Initialize LuminaLink
        
        Args:
            video_path: Video file path
            characters_config: Character config file path
            preview: Enable preview mode
            output_subtitles: Subtitle output path
        """
        self.video_path = video_path
        self.characters_config = characters_config
        self.preview = preview
        self.output_subtitles = output_subtitles
        
        self.config = get_config()
        
        # Initialize components
        self.video_processor: Optional[VideoProcessor] = None
        self.audio_detector: Optional[AudioDetector] = None
        self.character_recognizer: Optional[CharacterRecognizer] = None
        self.character_tracker: Optional[CharacterTracker] = None
        self.scene_analyzer: Optional[SceneAnalyzer] = None
        self.narrator: Optional[Narrator] = None
        self.tts_manager: Optional[TTSManager] = None
        self.narration_player: Optional[NarrationPlayer] = None
        
        # Running state
        self._running = False
        self._paused = False
    
    def setup(self):
        """Initialize all components"""
        console.print(Panel.fit(
            "[bold blue]LuminaLink - AI Movie Audio Description[/bold blue]\n"
            "Initializing...",
            border_style="blue"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            # 1. Initialize video processor
            task = progress.add_task("[cyan]Loading video...", total=6)
            self.video_processor = VideoProcessor(self.video_path)
            if not self.video_processor.open():
                raise RuntimeError(f"Failed to open video: {self.video_path}")
            progress.update(task, advance=1)
            
            # 2. Extract audio
            progress.update(task, description="[cyan]Analyzing audio...")
            self.audio_detector = AudioDetector(
                silence_threshold_db=self.config.narration.silence_threshold,
                min_silence_duration=1.5,
                min_narration_gap=self.config.narration.interval
            )
            self.audio_detector.load_audio_from_video(self.video_path)
            progress.update(task, advance=1)
            
            # 3. Initialize character recognition
            progress.update(task, description="[cyan]Loading character recognition module...")
            self.character_recognizer = CharacterRecognizer()
            if self.characters_config:
                self._load_characters_config()
            self.character_tracker = CharacterTracker(self.character_recognizer)
            progress.update(task, advance=1)
            
            # 4. Initialize scene analyzer
            progress.update(task, description="[cyan]Initializing AI model...")
            self.scene_analyzer = SceneAnalyzer()
            # Set known characters
            known_chars = [c.name for c in self.character_recognizer.list_characters()]
            self.scene_analyzer.set_characters(known_chars)
            progress.update(task, advance=1)
            
            # 5. Initialize narration generator
            progress.update(task, description="[cyan]Initializing narration module...")
            self.narrator = Narrator(style=NarrationStyle.CONCISE)
            progress.update(task, advance=1)
            
            # 6. Initialize TTS
            progress.update(task, description="[cyan]Initializing speech synthesis...")
            self.tts_manager = TTSManager()
            self.narration_player = NarrationPlayer(self.tts_manager)
            progress.update(task, advance=1)
        
        # Display video info
        video_info = self.video_processor.get_info()
        if video_info:
            table = Table(title="Video Info", show_header=False)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("File", Path(video_info.path).name)
            table.add_row("Resolution", f"{video_info.width}x{video_info.height}")
            table.add_row("Frame Rate", f"{video_info.fps:.2f} FPS")
            table.add_row("Duration", f"{video_info.duration / 60:.1f} minutes")
            console.print(table)
        
        console.print("[green]Initialization complete![/green]\n")
    
    def _load_characters_config(self):
        """Load character configuration"""
        import json
        
        config_path = Path(self.characters_config)
        if not config_path.exists():
            logger.warning(f"Character config file not found: {config_path}")
            return
        
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for char_data in data.get("characters", []):
            self.character_recognizer.add_character(
                name=char_data["name"],
                aliases=char_data.get("aliases", []),
                description=char_data.get("description", "")
            )
        
        logger.info(f"Loaded {len(data.get('characters', []))} character configs")
    
    async def run(self):
        """Run the narration system"""
        self.setup()
        
        self._running = True
        
        console.print("[bold]Starting video analysis...[/bold]")
        console.print("Tip: Press Ctrl+C to stop\n")
        
        try:
            # Analyze audio, get narration time slots
            audio_segments = self.audio_detector.analyze_audio()
            silence_windows = self.audio_detector.find_silence_windows(audio_segments)
            narration_slots = self.audio_detector.get_narration_slots(silence_windows)
            
            console.print(f"[cyan]Found {len(narration_slots)} narration opportunities[/cyan]\n")
            
            # Process each narration time slot
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                
                task = progress.add_task(
                    "[cyan]Processing video...",
                    total=len(narration_slots)
                )
                
                for i, (start_time, end_time) in enumerate(narration_slots):
                    if not self._running:
                        break
                    
                    progress.update(
                        task,
                        description=f"[cyan]Processing {start_time:.1f}s - {end_time:.1f}s"
                    )
                    
                    # Get frame at this time point
                    frame = self.video_processor.get_frame_at(start_time)
                    
                    if frame is None:
                        progress.update(task, advance=1)
                        continue
                    
                    # Recognize characters
                    characters = self.character_recognizer.get_characters_in_frame(
                        frame.frame,
                        timestamp=start_time
                    )
                    
                    # Analyze scene
                    scene_analysis = await self.scene_analyzer.analyze_frame_async(
                        frame.frame,
                        characters_in_frame=characters,
                        timestamp=start_time
                    )
                    
                    # Generate narration
                    narration = self.narrator.generate_narration(
                        scene_analysis,
                        slot=(start_time, end_time),
                        characters_in_frame=characters
                    )
                    
                    if narration:
                        # Display narration content
                        console.print(narration.text, style="green")
                        
                        # Pre-generate audio
                        await self.narration_player.prepare_narration(
                            narration.text,
                            narration_id=f"narration_{i}"
                        )
                    
                    progress.update(task, advance=1)
            
            # Export subtitles
            if self.output_subtitles:
                self.narrator.export_subtitles(self.output_subtitles)
                console.print(f"\n[green]Subtitles saved to: {self.output_subtitles}[/green]")
            
            # Display statistics
            self._show_statistics()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]User interrupted[/yellow]")
        finally:
            self._running = False
            self.cleanup()
    
    def _show_statistics(self):
        """Display statistics"""
        history = self.narrator.get_history()
        
        console.print("\n")
        table = Table(title="Narration Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Narrations", str(len(history)))
        
        if history:
            total_duration = sum(n.duration for n in history)
            table.add_row("Total Duration", f"{total_duration:.1f} seconds")
            
            total_chars = sum(len(n.text) for n in history)
            table.add_row("Total Characters", str(total_chars))
        
        # Character statistics
        characters = self.character_recognizer.list_characters()
        if characters:
            table.add_row("Recognized Characters", str(len(characters)))
            top_char = max(characters, key=lambda c: c.appearance_count)
            table.add_row("Most Frequent Character", f"{top_char.name} ({top_char.appearance_count} times)")
        
        console.print(table)
    
    def cleanup(self):
        """Clean up resources"""
        if self.video_processor:
            self.video_processor.close()
        
        if self.narration_player:
            self.narration_player.clear_cache()
        
        if self.character_recognizer:
            self.character_recognizer.save_characters()
        
        console.print("[dim]Resources cleaned up[/dim]")
    
    def stop(self):
        """Stop running"""
        self._running = False
    
    def pause(self):
        """Pause"""
        self._paused = True
    
    def resume(self):
        """Resume"""
        self._paused = False


@click.group()
def cli():
    """
    LuminaLink - AI Movie Audio Description for the Visually Impaired

    Provides real-time movie narration for visually impaired viewers
    """
    pass


@cli.command()
@click.option(
    "--video", "-v",
    required=True,
    type=click.Path(exists=True),
    help="Video file path"
)
@click.option(
    "--characters", "-c",
    type=click.Path(exists=True),
    help="Character config file path (JSON)"
)
@click.option(
    "--preview", "-p",
    is_flag=True,
    help="Enable preview mode"
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Output subtitle file path (.srt)"
)
@click.option(
    "--register-characters",
    is_flag=True,
    help="Start character registration mode"
)
def process(
    video: str,
    characters: Optional[str],
    preview: bool,
    output: Optional[str],
    register_characters: bool
):
    """
    Process video file and generate narration subtitles (file-based mode)
    """
    if register_characters:
        # TODO: Implement character registration mode
        console.print("[yellow]Character registration mode not yet implemented[/yellow]")
        return

    app = LuminaLink(
        video_path=video,
        characters_config=characters,
        preview=preview,
        output_subtitles=output
    )

    asyncio.run(app.run())


@cli.command()
@click.option(
    "--camera", "-c",
    default=0,
    type=int,
    help="Camera device index (default: 0)"
)
@click.option(
    "--characters",
    type=click.Path(exists=True),
    help="Character config file path (JSON)"
)
@click.option(
    "--width",
    default=1280,
    type=int,
    help="Camera resolution width (default: 1280)"
)
@click.option(
    "--height",
    default=720,
    type=int,
    help="Camera resolution height (default: 720)"
)
@click.option(
    "--fps",
    default=30,
    type=int,
    help="Camera frame rate (default: 30)"
)
def camera(
    camera: int,
    characters: Optional[str],
    width: int,
    height: int,
    fps: int
):
    """
    Real-time camera narration mode with GUI

    Opens a GUI window that displays the camera feed and provides
    real-time narration of the captured video. Automatically pauses
    narration during dialogue.
    """
    console.print(Panel.fit(
        "[bold blue]LuminaLink Camera Mode[/bold blue]\n"
        "Starting GUI...",
        border_style="blue"
    ))

    try:
        from .gui.camera_app import CameraApp

        app = CameraApp(
            camera_index=camera,
            characters_config=characters
        )

        console.print("[green]✓ GUI initialized[/green]")
        console.print(f"[cyan]Camera: {camera} | Resolution: {width}x{height} @ {fps}fps[/cyan]")
        console.print("[dim]Use Space to pause/resume, Esc to stop[/dim]\n")

        app.run()

    except ImportError as e:
        console.print(f"[red]✗ Failed to import GUI module: {e}[/red]")
        console.print("[yellow]Please ensure all dependencies are installed:[/yellow]")
        console.print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Camera mode error: {e}", exc_info=True)
        console.print(f"[red]✗ Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
