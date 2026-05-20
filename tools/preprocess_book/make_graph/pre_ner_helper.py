from __future__ import annotations

from typing import Any, Dict, List, Sequence


def build_required_entities_from_chapter_blocks(
    chapter_blocks: Sequence[Dict[str, Any]],
    max_entities_per_batch: int = 120,
) -> List[Dict[str, Any]]:
    """Pre-NER helper is hard-disabled in current pipeline settings.

    Stub kept to preserve import compatibility when PRE_NER is disabled.
    """
    return []

