import json
from pathlib import Path
from typing import Optional

import typer

from luminalink.config import AppConfig
from luminalink.pipeline.engine import NarrationEngine

app = typer.Typer(add_completion=False)


@app.command()
def run(
    video: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    config: Optional[Path] = typer.Option(None, exists=True, file_okay=True, dir_okay=False),
    output_jsonl: Optional[Path] = typer.Option(None, file_okay=True, dir_okay=False),
    character_db: Optional[Path] = typer.Option(None, file_okay=True, dir_okay=False),
) -> None:
    """Run phase-1 vision-to-text narration on a local video file."""

    cfg = AppConfig.load(config)
    if character_db is not None:
        cfg.character_db.path = str(character_db)

    engine = NarrationEngine(cfg)
    events = engine.run_video_file(video_path=str(video), output_jsonl_path=str(output_jsonl) if output_jsonl else None)

    if output_jsonl is None:
        for event in events:
            typer.echo(json.dumps(event.model_dump(), ensure_ascii=False))


@app.command()
def init_character_db(
    db_path: Path = typer.Option(..., file_okay=True, dir_okay=False),
) -> None:
    """Create (or migrate) the character database."""

    from luminalink.store.character_store import CharacterStore

    store = CharacterStore(str(db_path))
    store.migrate()
    typer.echo(f"Initialized character DB at: {db_path}")


@app.command()
def add_character(
    db_path: Path = typer.Option(..., file_okay=True, dir_okay=False),
    name: str = typer.Option(...),
    face_image: Optional[Path] = typer.Option(None, exists=True, file_okay=True, dir_okay=False),
) -> None:
    """Add a character and optionally enroll face embedding from an image."""

    from luminalink.store.character_store import CharacterStore
    from luminalink.vision.face_pipeline import FacePipeline

    store = CharacterStore(str(db_path))
    store.migrate()
    character_id = store.create_character(display_name=name)

    if face_image is not None:
        pipeline = FacePipeline()
        embedding = pipeline.embedding_from_image_path(str(face_image))
        store.add_embedding(character_id=character_id, model_name=pipeline.model_name, vector=embedding)

    typer.echo(f"Added character: {name} ({character_id})")


@app.command()
def self_check() -> None:
    """Run lightweight self-checks that don't require heavyweight ML dependencies."""

    from luminalink.selfcheck import run_self_check

    run_self_check()

