from __future__ import annotations

from abc import ABC, abstractmethod

from luminalink.config import NLGConfig
from luminalink.types import VisualFacts


class NarrationGenerator(ABC):
    """Abstract interface for narration text generation."""

    @abstractmethod
    def generate(self, facts: VisualFacts, cfg: NLGConfig) -> tuple[str, float]:
        """Generate narration text and a confidence score."""


class RuleBasedNarrationGenerator(NarrationGenerator):
    """A lightweight generator that uses deterministic templates."""

    def generate(self, facts: VisualFacts, cfg: NLGConfig) -> tuple[str, float]:
        """Generate narration text from VisualFacts."""

        names = [c.display_name for c in facts.characters[:2]]
        objs = [o.label for o in facts.objects[: max(1, cfg.detail_level)]]

        subject = ", ".join(names) if names else "The scene"
        if objs:
            obj_text = ", ".join(objs)
            text = f"{subject} shows {obj_text}."
        else:
            text = f"{subject} changes." if facts.scene_change else f"{subject} continues."
        return text, 0.55

