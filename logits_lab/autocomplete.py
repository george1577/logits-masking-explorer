"""Autocomplete grounded on a document: a person acts as the sequence selector."""


def load_document(path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def grounding_lines(document: list[str], typed_prefix: str, limit: int = 40) -> list[str]:
    """Lines of the document containing the typed prefix (case-insensitive), at most `limit`."""
    needle = typed_prefix.strip().lower()
    if not needle:
        return []
    return [line for line in document if needle in line.lower()][:limit]


def autocomplete_messages(examples: list[str], typed_prefix: str) -> list[dict]:
    examples_block = "\n".join(examples) if examples else "(no matching lines)"
    system = (
        "Complete this phrase in a style similar to the ones below. You are acting as auto-complete.\n"
        "Simply complete the phrase without any introduction or preamble.\n"
        "Make sure it is only one sentence.\n\n"
        f"** Examples **:\n{examples_block}"
    )
    user = f"** Phrase **:\n{typed_prefix} ____"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
