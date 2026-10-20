"""Small, deterministic lexical scoring for transcript words.

This module deliberately uses only the transcript text.  It does not infer
topics or meaning with a semantic or AI model; it ranks normalized tokens by
simple lexical properties instead.
"""

from __future__ import annotations

import operator
import re
import unicodedata


# This is intentionally conservative: common grammatical words and spoken
# hesitation/discourse fillers are excluded, while words such as ``like`` are
# retained because they can carry meaning in a transcript.
_STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does doing
    for from had has have he her here hers him his how i if in into is it its
    me more most my no not of on or our ours she so some than that the their
    theirs them then there these they this those to too us very was we were
    what when where which while who will with would you your yours
    actually basically literally really okay ok yeah yes
    ah ahh er erm hmm mm mm-hm uh uhh um umm
    """.split()
)
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(value: object) -> list[tuple[str, str]]:
    """Return ``(normalized, source)`` tokens from one text value."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    result = []
    for match in _TOKEN.finditer(text):
        source = match.group(0)
        normalized = source.casefold()
        if normalized:
            result.append((normalized, source))
    return result


def _explicit_tokens(explicit: object) -> list[str]:
    """Normalize explicit terms, accepting one string as one term."""

    values = (explicit,) if isinstance(explicit, str) else (explicit or ())
    result = []
    seen = set()
    for value in values:
        for token, _ in _tokens(value):
            if token not in seen:
                seen.add(token)
                result.append(token)
    return result


def _is_acronym(source: str) -> bool:
    """Whether the source spelling is an uppercase, multi-letter acronym."""

    letters = [character for character in source if character.isalpha()]
    return len(letters) >= 2 and all(character.isupper() for character in letters)


def score_important_words(words, *, max_words=3, explicit=()) -> tuple[str, ...]:
    """Return up to ``max_words`` normalized, lexically important tokens.

    ``words`` may contain any objects exposing ``.text``; optional timing
    attributes are ignored.  Explicit terms are normalized and placed first,
    including terms that are absent from the transcript.  Remaining terms are
    selected using deterministic lexical signals: numbers and uppercase
    acronyms receive a boost, followed by token length and occurrence count.
    Selected transcript terms are emitted in their first transcript order.

    This is a transparent lexical heuristic, not semantic AI understanding.
    """

    try:
        limit = operator.index(max_words)
    except TypeError as exc:
        raise TypeError("max_words must be an integer") from exc
    if limit < 0:
        raise ValueError("max_words must be non-negative")
    if not limit:
        return ()

    explicit_tokens = _explicit_tokens(explicit)
    explicit_set = set(explicit_tokens)

    occurrences: dict[str, list[str]] = {}
    order: dict[str, int] = {}
    position = 0
    for word in words or ():
        for token, source in _tokens(getattr(word, "text", "")):
            if token in _STOPWORDS:
                position += 1
                continue
            if token not in occurrences:
                occurrences[token] = [source]
                order[token] = position
            else:
                occurrences[token].append(source)
            position += 1

    ranked = []
    for token, sources in occurrences.items():
        if token in explicit_set:
            continue
        category = 2 if any(character.isdigit() for character in token) else 1
        if any(_is_acronym(item) for item in sources):
            category = 2
        # The tuple is sorted descending below.  First appearance is only a
        # final tie-breaker; output order is rebuilt separately afterward.
        ranked.append((category, len(token), len(sources), -order[token], token))
    ranked.sort(reverse=True)

    selected = explicit_tokens[:limit]
    remaining = limit - len(selected)
    if remaining > 0:
        selected_set = set(selected)
        selected.extend(token for _, _, _, _, token in ranked
                        if token not in selected_set)
        del selected[limit:]

    # Explicit terms retain their requested order.  Transcript-derived terms
    # retain first appearance order, even though ranking chose the shortlist.
    explicit_part = selected[:len(explicit_tokens)]
    transcript_part = selected[len(explicit_tokens):]
    transcript_part.sort(key=order.__getitem__)
    return tuple(explicit_part + transcript_part)


__all__ = ["score_important_words"]
