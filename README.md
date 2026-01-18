
# LuminaLink (Phase 1: Vision-to-Text)

This repository scaffolds a **modular movie narration system for blind users**.

## Quick start

```bash
uv sync
luminalink self-check
luminalink run --video .tmp/sample.mp4
```

## Optional extras

- Vision (PyTorch/TorchVision + facenet-pytorch):

```bash
uv sync --extra vision
```

- NLG (TensorFlow):

```bash
uv sync --extra nlg
```

## Character database

```bash
luminalink init-character-db --db-path data/characters.sqlite
luminalink add-character --db-path data/characters.sqlite --name "乔" --face-image /path/to/joe.jpg
```

Then run:

```bash
luminalink run --video /path/to/movie.mp4 --character-db data/characters.sqlite --output-jsonl out/narration.jsonl
```

## Docs

- [PRD](docs/PRD.md)
- [ERD](docs/ERD.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Interfaces](docs/INTERFACES.md)
