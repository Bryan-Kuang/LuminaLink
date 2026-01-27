from __future__ import annotations

from luminalink.config import NLGConfig
from luminalink.nlg.generator import NarrationGenerator
from luminalink.types import VisualFacts


class TensorFlowTemplateNarrator(NarrationGenerator):
    """A TensorFlow-based template narrator (no training required)."""

    def __init__(self):
        try:
            import tensorflow as tf
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("TensorFlow narrator requires extras: `uv sync --extra nlg`") from e
        self._tf = tf

    def generate(self, facts: VisualFacts, cfg: NLGConfig) -> tuple[str, float]:
        """Generate narration using `tf.strings` operations for easy later upgrades."""

        tf = self._tf
        names = [c.display_name for c in facts.characters[:2]]
        objs = [o.label for o in facts.objects[: max(1, cfg.detail_level)]]

        names_t = tf.constant(names, dtype=tf.string)
        objs_t = tf.constant(objs, dtype=tf.string)

        subject = tf.cond(
            tf.size(names_t) > 0,
            lambda: tf.strings.reduce_join(names_t, separator=", "),
            lambda: tf.constant("The scene", dtype=tf.string),
        )
        obj_text = tf.cond(
            tf.size(objs_t) > 0,
            lambda: tf.strings.reduce_join(objs_t, separator=", "),
            lambda: tf.constant("", dtype=tf.string),
        )

        if cfg.style == "detailed":
            template = tf.constant("{subject} shows {objects}, with clear details.", dtype=tf.string)
        else:
            template = tf.constant("{subject} shows {objects}.", dtype=tf.string)

        text_t = tf.cond(
            tf.strings.length(obj_text) > 0,
            lambda: tf.strings.format(template, {"subject": subject, "objects": obj_text}),
            lambda: tf.strings.format("{subject} continues.", {"subject": subject}),
        )
        text = str(text_t.numpy().decode("utf-8"))
        return text, 0.65

