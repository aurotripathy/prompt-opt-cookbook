"""Local utilities for the NPS notebook."""

from __future__ import annotations

import textwrap
from collections.abc import Iterable
from typing import Any

__all__ = ["wrap_text", "list_exact_match_raw", "list_exact_match_strict"]


def wrap_text(text: str, width: int = 80) -> str:
    """Wrap ``text`` to ``width`` columns, preserving paragraph and line breaks.

    Paragraphs (separated by a blank line) are wrapped independently. Within a
    paragraph that already contains explicit line breaks (e.g. a block of
    equations or list items), each line is wrapped on its own so the structure
    survives the reflow.

    Args:
        text: Source text. May contain ``\\n`` (line breaks) and ``\\n\\n``
            (paragraph breaks).
        width: Maximum line width in columns. Defaults to ``80``.

    Returns:
        The wrapped string. Paragraph and intra-paragraph line breaks from the
        input are preserved.
    """
    out_paragraphs: list[str] = []
    for paragraph in text.split("\n\n"):
        if "\n" in paragraph:
            lines = [
                textwrap.fill(line, width=width) if len(line) > width else line
                for line in paragraph.split("\n")
            ]
            out_paragraphs.append("\n".join(lines))
        else:
            out_paragraphs.append(textwrap.fill(paragraph, width=width))
    return "\n\n".join(out_paragraphs)


def list_exact_match_raw(gold: Any, predicted: Any) -> float:
    """Per-row accuracy for list-valued labels, treating order as irrelevant.

    Returns ``1.0`` iff ``gold`` and ``predicted`` contain the same elements
    (as sets), else ``0.0``. Designed for column-wise comparison with
    ``list(map(list_exact_match_raw, df.gold_col, df.pred_col))``.

    A ``None`` (or any non-iterable) on either side scores ``0.0`` rather than
    raising, so missing predictions don't blow up a whole evaluation pass.
    """
    if not isinstance(gold, Iterable) or not isinstance(predicted, Iterable):
        return 0.0
    if isinstance(gold, str) or isinstance(predicted, str):
        return 1.0 if gold == predicted else 0.0
    return 1.0 if set(gold) == set(predicted) else 0.0


def list_exact_match_strict(gold: Any, predicted: Any) -> float:
    """Strict per-row accuracy for list-valued labels.

    Returns ``1.0`` iff ``gold`` and ``predicted`` are equal as ordered
    sequences (so ``[a, b] != [b, a]``). Use this when order or duplicates
    carry meaning; otherwise prefer :func:`list_exact_match_raw`.
    """
    if gold is None or predicted is None:
        return 0.0
    return 1.0 if list(gold) == list(predicted) else 0.0
