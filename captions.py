"""Dependency-free, output-timeline ASS captions for FFmpeg/libass.

The looks are original charcoal/mint treatments with off-white text. The
Premiere-inspired option is only a generic editorial fade/slide approximation;
it uses no Adobe assets. Keyword selection is a simple lexical heuristic, not
semantic AI. Fonts must be installed on the rendering machine.

Words, when present, are authoritative (including their already-remapped times).
Empty word lists fall back to evenly timed tokens from segment.text. Invalid or
nonpositive intervals are omitted; positive sub-centisecond intervals occupy one
ASS tick. Display holds are capped at four seconds. Literal braces/backslashes
are replaced by full-width Unicode equivalents, and whitespace is flattened, so
untrusted transcript text can never create ASS override tags or line breaks.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


def _preset(label, description, animation, fonts, primary, accent, outline,
            background, color_grade_guidance, **style):
    """Build one serializable preset record.

    Colors intentionally use ASS's native ``&HAABBGGRR&`` notation.  Keeping
    the values in the metadata makes the generated style and important-word
    tags use exactly the same palette, without depending on an installed font
    or an external template.
    """
    style.setdefault("style_id", label.lower().replace(" ", "-").replace("/", "-"))
    style.setdefault("panel_style", "box" if style.get("border", 1) == 3 else "outline")
    style.setdefault("panel_opacity", 0.31 if style["panel_style"] == "box" else 0.0)
    style.setdefault("safe_zone", "lower-third")
    style.setdefault("easing", "ease-out")
    style.setdefault("animation_in", animation)
    style.setdefault("animation_out", animation)
    style.setdefault("emphasis_style", "accent-bold")
    return {
        "label": label,
        "description": description,
        "animation": animation,
        "font_family_candidates": tuple(fonts),
        # The shorter spelling is useful to callers that present metadata in
        # their own UI; keep the canonical, explicit name above as well.
        "font_candidates": tuple(fonts),
        "primary_color": primary,
        "accent_color": accent,
        "outline_color": outline,
        "background_color": background,
        "color_grade_guidance": color_grade_guidance,
        "primary_hex": style.pop("primary_hex", ""),
        "accent_hex": style.pop("accent_hex", ""),
        "outline_hex": style.pop("outline_hex", ""),
        "background_hex": style.pop("background_hex", ""),
        **style,
    }


# ASS/libass has a small, portable animation vocabulary.  These are original
# timing recipes rather than imported templates.  Keeping the catalog public
# lets a UI show meaningful choices without duplicating renderer knowledge.
ANIMATION_METADATA = {
    "none": {"label": "None", "easing": "linear", "in_ms": 0, "out_ms": 0},
    "fade": {"label": "Fade", "easing": "ease-in-out", "in_ms": 150, "out_ms": 150},
    "slide": {"label": "Slide", "easing": "ease-out", "in_ms": 150, "out_ms": 150},
    "pop": {"label": "Pop", "easing": "ease-out", "in_ms": 150, "out_ms": 150},
    "blur": {"label": "Blur", "easing": "ease-in-out", "in_ms": 150, "out_ms": 150},
    "wipe": {"label": "Wipe", "easing": "ease-out", "in_ms": 180, "out_ms": 180},
    "spring": {"label": "Spring", "easing": "spring", "in_ms": 220, "out_ms": 180},
    "rise": {"label": "Rise", "easing": "ease-out", "in_ms": 180, "out_ms": 150},
    "drift": {"label": "Drift", "easing": "ease-in-out", "in_ms": 220, "out_ms": 220},
    "typewriter": {"label": "Typewriter", "easing": "linear", "in_ms": 0, "out_ms": 150},
}
ANIMATIONS = tuple(ANIMATION_METADATA)


# These are original, generic visual directions.  The names describe the
# editing intent only; they do not represent official creator or software
# templates and contain no copied assets.
_DEFAULT_PRIMARY = "&H00F4F5F3&"
_DEFAULT_ACCENT = "&H00BCDFA0&"  # #A0DFBC, expressed as ASS BBGGRR.
_DEFAULT_OUTLINE = "&H001C2028&"  # #28201C, expressed as ASS BBGGRR.
_DEFAULT_BACKGROUND = "&H501C2028&"
_DEFAULT_FONTS = ("Segoe UI", "Arial", "Noto Sans")


PRESETS = {
    "alex_hormozi": _preset(
        "High-contrast impact", "Original bold impact treatment with compact scale motion.",
        "pop", ("Montserrat Black", "Anton", "Montserrat", "Segoe UI"), "&H00FFFFFF&", "&H0000E5FF&",
        "&H00000000&", "&H80000000&", "Keep skin tones neutral; protect bright yellow from clipping.",
        primary_hex="#FFFFFF", accent_hex="#FFE500", outline_hex="#000000", background_hex="#000000",
        bold=-1, spacing=0.8, border=1, outline_scale=0.075),
    "mrbeast": _preset(
        "Bright challenge", "Original energetic yellow accent with a fast scale pulse.",
        "pop", ("Luckiest Guy", "Bangers", "Arial", "Segoe UI"), "&H00FFFFFF&", "&H0000D6FF&",
        "&H00000000&", "&H80000000&", "Use a clean, high-key grade with saturated yellow accents.",
        primary_hex="#FFFFFF", accent_hex="#FFD600", accent_palette_hex=("#FFD600", "#00D9FF", "#B9FF00"), outline_hex="#000000", background_hex="#000000",
        bold=-1, spacing=0.4, border=1, outline_scale=0.07),
    "karaoke": _preset(
        "Karaoke", "An original timed-word treatment that accents the active word in mint.",
        "none", ("Poppins ExtraBold", "Montserrat Bold", "Poppins", "Segoe UI"), "&H00FFFFFF&", "&H0000FFB6&", "&H00000000&",
        _DEFAULT_BACKGROUND, "Keep contrast neutral and let the mint accent carry timing.",
        primary_hex="#FFFFFF", accent_hex="#B6FF00", accent_palette_hex=("#B6FF00", "#FFEB00"), outline_hex="#000000", background_hex="#28201C"),
    "iman_gadzhi": _preset(
        "Warm authority", "Original restrained business-caption treatment with a gold accent.",
        "fade", ("Montserrat Medium", "Montserrat Bold", "Segoe UI"), "&H00E6F0F5&", "&H0037AFD4&",
        "&H001A1714&", "&H701A1714&", "Warm the mids slightly while keeping gold highlights controlled.",
        primary_hex="#F5F0E6", accent_hex="#D4AF37", outline_hex="#14171A", background_hex="#14171A",
        bold=-1, outline_scale=0.05),
    "ali_abdaal": _preset(
        "Clean study", "Original friendly educational captions with a calm blue highlight.",
        "fade", ("Inter SemiBold", "Poppins Medium", "Inter", "Segoe UI"), "&H00FFFFFF&", "&H0077CB6B&",
        "&H00251A12&", "&H70251A12&", "Favor a bright, soft grade with gentle blue contrast.",
        primary_hex="#FFFFFF", accent_hex="#6BCB77", outline_hex="#121A25", background_hex="#121A25"),
    "devin_jatho": _preset(
        "Founder focus", "Original direct founder-style captions with a warm orange accent.",
        "pop", ("Montserrat ExtraBold", "Tactic Sans Bold Italic", "Montserrat", "Segoe UI"), "&H00FFFFFF&", "&H00FFEDB7&",
        "&H00100B08&", "&H80100B08&", "Preserve neutral blacks and a crisp orange highlight.",
        primary_hex="#FFFFFF", accent_hex="#B7EDFF", outline_hex="#080B10", background_hex="#080B10",
        bold=-1, outline_scale=0.06),
    "luxury_editorial": _preset(
        "Luxury editorial", "Original quiet editorial typography with a measured gold accent.",
        "fade", ("Playfair Display", "Montserrat Medium", "Georgia", "Segoe UI"), "&H00E8F2F7&", "&H007AB7D6&",
        "&H00110E0A&", "&H80110E0A&", "Use low-saturation shadows and reserve gold for key words.",
        primary_hex="#F7F2E8", accent_hex="#D6B77A", outline_hex="#0A0E11", background_hex="#0A0E11",
        outline_scale=0.04),
    "tiktok_native_box": _preset(
        "Native box", "Original compact boxed captions with a quick social-video pop.",
        "pop", ("TikTok Sans", "Arial Bold", "Roboto Medium", "Segoe UI"), "&H00FFFFFF&", "&H00FFFFFF&",
        "&H00101010&", "&H90101010&", "Use a punchy, high-contrast grade and avoid crushed faces.",
        primary_hex="#FFFFFF", accent_hex="#FFFFFF", outline_hex="#101010", background_hex="#101010",
        bold=-1, border=3, outline_scale=0.035),
    "modern_podcast": _preset(
        "Modern podcast", "Original conversational captions with a soft, readable entrance.",
        "fade", ("Montserrat Bold", "Inter Medium", "Inter", "Segoe UI"), "&H00FFFFFF&", "&H0000DDFF&",
        "&H00221A14&", "&H70221A14&", "Keep voices natural with a mild warm-neutral grade.",
        primary_hex="#FFFFFF", accent_hex="#FFDD00", outline_hex="#141A22", background_hex="#141A22"),
    "documentary_cinematic": _preset(
        "Documentary cinematic", "Original understated documentary captions with a slow editorial fade.",
        "fade", ("Helvetica Neue", "Source Sans 3", "Inter Medium", "Merriweather"), "&H00E9F0F2&", "&H00B3D5C2&",
        "&H00201E1A&", "&H70201E1A&", "Protect highlight detail and keep the palette subdued and filmic.",
        primary_hex="#F2F0E9", accent_hex="#B3D5C2", outline_hex="#1A1E20", background_hex="#1A1E20"),
    "news_explainer": _preset(
        "News explainer", "Original information-first captions with a precise slide transition.",
        "slide", ("Archivo Black", "Roboto Medium", "Roboto", "Segoe UI"), "&H00FFFFFF&", "&H002828E6&",
        "&H0020160E&", "&H8020160E&", "Use a neutral broadcast grade with clear orange signal accents.",
        primary_hex="#FFFFFF", accent_hex="#E62828", accent_palette_hex=("#E62828", "#102C46"), outline_hex="#102C46", background_hex="#102C46",
        border=1),
    "dark_motivational": _preset(
        "Dark motivational", "Original high-contrast motivational captions with a resolving blur.",
        "blur", ("Bebas Neue", "Montserrat SemiBold", "Montserrat", "Segoe UI"), "&H00FFFFFF&", "&H00006AFF&",
        "&H000A0705&", "&H900A0705&", "Favor deep shadows while retaining texture in dark footage.",
        primary_hex="#FFFFFF", accent_hex="#FF6A00", outline_hex="#000000", background_hex="#05070A",
        bold=-1, outline_scale=0.065),
    "neon_gaming": _preset(
        "Neon gaming", "Original neon-accented captions with crisp pop timing.",
        "pop", ("Anton", "Rajdhani Bold", "Consolas", "Segoe UI"), "&H00FFFFFF&", "&H00FFE500&",
        "&H0020140C&", "&H90101420&", "Keep blacks cool and let cyan-magenta highlights remain luminous.",
        primary_hex="#FFFFFF", accent_hex="#00E5FF", accent_palette_hex=("#00E5FF", "#A855F7", "#B6FF00"), outline_hex="#0C1420", background_hex="#201420",
        bold=-1, spacing=0.5, outline_scale=0.07),
    "retro_vhs": _preset(
        "Retro VHS", "Original analog-inspired caption color and slide approximation.",
        "slide", ("Archivo Black", "Space Mono", "Courier New", "Consolas"), "&H00CEF0FF&", "&H004370FF&",
        "&H00341B1A&", "&H80341B1A&", "Use restrained chroma and mild warm/cool separation; avoid fake scan assets.",
        primary_hex="#FFF0CE", accent_hex="#FF7043", accent_palette_hex=("#FF7043", "#FFF0CE"), outline_hex="#1A1B34", background_hex="#341B34",
        spacing=0.6),
    "handwritten_vlog": _preset(
        "Handwritten vlog", "Original casual typewriter build using installed handwriting-capable fonts when available.",
        "typewriter", ("Permanent Marker", "Poppins Medium", "Caveat Bold", "DM Sans", "Segoe UI"), "&H00EAF8FF&", "&H008AB1A3&",
        "&H00201813&", "&H70201813&", "Keep the grade light and personal with a small coral signal accent.",
        primary_hex="#FFF8EA", accent_hex="#A3B18A", accent_palette_hex=("#A3B18A", "#C77D58"), outline_hex="#131820", background_hex="#131820"),
    # Legacy names remain first-class keys so callers do not need a migration.
    "clean": _preset(
        "Clean", "Off-white type, charcoal edge, and a gentle fade.", "fade", _DEFAULT_FONTS,
        _DEFAULT_PRIMARY, _DEFAULT_ACCENT, _DEFAULT_OUTLINE, _DEFAULT_BACKGROUND,
        "Keep the grade neutral and preserve readable midtone contrast.",
        primary_hex="#F3F5F4", accent_hex="#A0DFBC", outline_hex="#28201C", background_hex="#28201C"),
    "punch": _preset(
        "Punch", "Bold, spaced lettering with a compact scale pulse.", "pop", _DEFAULT_FONTS,
        _DEFAULT_PRIMARY, _DEFAULT_ACCENT, _DEFAULT_OUTLINE, _DEFAULT_BACKGROUND,
        "Use a clean high-contrast grade with a restrained mint signal.",
        primary_hex="#F3F5F4", accent_hex="#A0DFBC", outline_hex="#28201C", background_hex="#28201C",
        bold=-1, spacing=0.7),
    "marker": _preset(
        "Marker", "Off-white lettering on an original charcoal caption panel.", "fade", _DEFAULT_FONTS,
        _DEFAULT_PRIMARY, _DEFAULT_ACCENT, _DEFAULT_OUTLINE, _DEFAULT_BACKGROUND,
        "Keep the background panel distinct without crushing the image.",
        primary_hex="#F3F5F4", accent_hex="#A0DFBC", outline_hex="#28201C", background_hex="#28201C",
        border=3),
    "pop": _preset(
        "Pop", "Crisp scale-in and scale-out motion.", "pop", _DEFAULT_FONTS, _DEFAULT_PRIMARY,
        _DEFAULT_ACCENT, _DEFAULT_OUTLINE, _DEFAULT_BACKGROUND,
        "Use a neutral grade and keep the mint pop legible.",
        primary_hex="#F3F5F4", accent_hex="#A0DFBC", outline_hex="#28201C", background_hex="#28201C",
        bold=-1),
    "slide": _preset(
        "Slide", "Short, lower-safe-area entrance and exit.", "slide", _DEFAULT_FONTS, _DEFAULT_PRIMARY,
        _DEFAULT_ACCENT, _DEFAULT_OUTLINE, _DEFAULT_BACKGROUND,
        "Preserve lower-frame detail while maintaining neutral contrast.",
        primary_hex="#F3F5F4", accent_hex="#A0DFBC", outline_hex="#28201C", background_hex="#28201C"),
    "typewriter": _preset(
        "Typewriter", "A phrase builds one timed word at a time.", "typewriter", _DEFAULT_FONTS,
        _DEFAULT_PRIMARY, _DEFAULT_ACCENT, _DEFAULT_OUTLINE, _DEFAULT_BACKGROUND,
        "Keep the image soft enough for the progressive build to read.",
        primary_hex="#F3F5F4", accent_hex="#A0DFBC", outline_hex="#28201C", background_hex="#28201C"),
    "blur": _preset(
        "Blur", "Soft focus resolves to clean text, then returns.", "blur", _DEFAULT_FONTS, _DEFAULT_PRIMARY,
        _DEFAULT_ACCENT, _DEFAULT_OUTLINE, _DEFAULT_BACKGROUND,
        "Use moderate contrast so blur transitions do not hide detail.",
        primary_hex="#F3F5F4", accent_hex="#A0DFBC", outline_hex="#28201C", background_hex="#28201C"),
    "premiere": _preset(
        "Premiere-inspired clean", "Original editorial fade/slide approximation; no Adobe assets.", "slide",
        _DEFAULT_FONTS, _DEFAULT_PRIMARY, _DEFAULT_ACCENT, _DEFAULT_OUTLINE, _DEFAULT_BACKGROUND,
        "Use a neutral editorial grade with restrained edge contrast.",
        primary_hex="#F3F5F4", accent_hex="#A0DFBC", outline_hex="#28201C", background_hex="#28201C"),
    "outline": _preset(
        "Outline", "Heavy charcoal outline for busy backgrounds.", "fade", _DEFAULT_FONTS, _DEFAULT_PRIMARY,
        _DEFAULT_ACCENT, _DEFAULT_OUTLINE, _DEFAULT_BACKGROUND,
        "Preserve background detail while keeping the outline visibly separated.",
        primary_hex="#F3F5F4", accent_hex="#A0DFBC", outline_hex="#28201C", background_hex="#28201C",
        bold=-1, outline_scale=0.075),
}

# Kept as module constants for callers that imported the old implementation's
# palette, and for the default legacy metadata above.
_WHITE = _DEFAULT_PRIMARY
_MINT = _DEFAULT_ACCENT
_CHARCOAL = _DEFAULT_OUTLINE
_STOPWORDS = frozenset("""about after again also because been before being between
could doing during every from have into just like more most much only other over
really should some than that their them then there these they thing things this
those through very want were what when where which while will with would your
you're you've that's there's don't doesn't didn't can't isn't wasn't aren't
the and for are but not you yes yeah okay um uh actually basically literally
""".split())
_TOKEN = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)


@dataclass(frozen=True)
class _Word:
    text: str
    start: int  # Centiseconds throughout event generation.
    end: int


def _plain(text) -> str:
    return " ".join(str(text or "").split())


def _escape(text: str) -> str:
    # ASS has no portable literal backslash/brace escape across all renderers.
    return _plain(text).translate(str.maketrans({"\\": "＼", "{": "｛", "}": "｝"}))


def _normalized(text: str) -> str:
    return " ".join(_TOKEN.findall(unicodedata.normalize("NFKC", text).casefold()))


def _tick(value) -> int:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Caption times must be finite")
    return max(0, math.floor(number * 100 + 0.500000001))


def _timed_tokens(text, start, end) -> list[_Word]:
    start_seconds, end_seconds = float(start), float(end)
    first, last = _tick(start_seconds), _tick(end_seconds)
    tokens = _plain(text).split()
    if not tokens or end_seconds <= max(0, start_seconds):
        return []
    last = max(first + 1, last)
    # Compound word records and fallback segments get deterministic timings.
    # When tokens outnumber available ticks, adjacent tokens share a tick.
    span = last - first
    return [
        _Word(token, first + span * i // len(tokens),
              max(first + span * i // len(tokens) + 1,
                  first + span * (i + 1) // len(tokens)))
        for i, token in enumerate(tokens)
    ]


def _words(segment) -> list[_Word]:
    supplied = list(getattr(segment, "words", None) or ())
    if supplied:
        result = [token for word in supplied
                  for token in _timed_tokens(word.text, word.start, word.end)]
    else:
        result = _timed_tokens(segment.text, segment.start, segment.end)
    return sorted(result, key=lambda word: (word.start, word.end))


def _units(text: str) -> float:
    """Conservative glyph estimate; no installed-font dependency required."""
    return sum(0 if unicodedata.combining(char) else
               1.05 if unicodedata.east_asian_width(char) in "WF" else
               1.03 if char in "WwMm@#%&" else
               0.38 if char.isspace() else 0.72 for char in _escape(text))


def _layout(words: list[_Word], size: float, available: float) -> tuple[int, float]:
    """Return a line-break index and a fit size (never more than two lines)."""
    texts = [word.text for word in words]
    if len(texts) <= 6 and _units(" ".join(texts)) * size <= available:
        return len(texts), size
    if len(texts) == 1:
        longest = _units(texts[0])
        return 1, min(size, available / max(1, longest))
    candidates = range(max(1, len(texts) - 6), min(6, len(texts) - 1) + 1)
    split = min(candidates, key=lambda i: (
        max(_units(" ".join(texts[:i])), _units(" ".join(texts[i:]))),
        abs(len(texts) - 2 * i)))
    longest = max(_units(" ".join(texts[:split])), _units(" ".join(texts[split:])))
    return split, min(size, available / max(1, longest))


def _groups(words: list[_Word], size: float, available: float):
    group = []
    for word in words:
        if group:
            candidate = group + [word]
            _, fit = _layout(candidate, size, available)
            if (len(group) >= 10 or word.end - group[0].start > 400
                    or word.start - group[-1].end > 75 or fit < size * 0.9
                    or (len(group) >= 4 and group[-1].text.endswith((".", "!", "?", ";")))):
                yield group
                group = []
        group.append(word)
    if group:
        yield group


def _stamp(ticks: int) -> str:
    seconds, fraction = divmod(ticks, 100)
    minutes, second = divmod(seconds, 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour}:{minute:02}:{second:02}.{fraction:02}"


def _motion(kind: str, duration: int, x: int, y: int, *, width: int = 1920,
            height: int = 1080) -> str:
    position = rf"\pos({x},{y})"
    # Do not spend a very short word's entire life invisible or scaled down.
    if kind == "none" or duration < 12:
        return position
    ms = duration * 10
    edge = min(150, ms // 4)
    out = ms - edge
    if kind == "fade":
        return position + rf"\fad({edge},{edge})"
    if kind == "pop":
        return (position + rf"\fscx88\fscy88\t(0,{edge},\fscx100\fscy100)"
                + rf"\t({out},{ms},\fscx88\fscy88)\fad({edge // 2},{edge // 2})")
    if kind == "blur":
        return (position + rf"\blur4\t(0,{edge},\blur0)"
                + rf"\t({out},{ms},\blur4)\fad({edge // 2},{edge // 2})")
    if kind == "wipe":
        # A clipped reveal/conceal is widely supported by libass and does not
        # require a raster asset.  The oversized bounds keep the sample useful
        # for both portrait and landscape frames.
        left, right = max(0, x - width // 2), min(width, x + width // 2)
        top, bottom = max(0, y - height // 5), min(height, y + height // 5)
        return (position + rf"\clip({left},{top},{left},{bottom})"
                + rf"\t(0,{edge},0.85,\clip({left},{top},{right},{bottom}))"
                + rf"\t({out},{ms},0.85,\clip({right},{top},{right},{bottom}))")
    if kind == "spring":
        return (position + rf"\fscx82\fscy82"
                + rf"\t(0,{edge},0.55,\fscx108\fscy108)"
                + rf"\t({edge},{min(ms, edge * 2)},0.7,\fscx100\fscy100)"
                + rf"\t({out},{ms},0.8,\fscx94\fscy94)\fad({edge // 2},{edge // 2})")
    if kind == "typewriter":
        return position + rf"\fad({edge // 2},{edge // 2})"
    # Slide is handled separately: libass permits only one \move per event.
    return position


def _travel_motion(kind: str, left: int, right: int, start: int, end: int,
                   x: int, y: int, distance: int) -> str:
    """Return a legal single-event move for an entrance or exit slice."""
    duration = max(1, right - left)
    edge = max(1, min(15, (end - start) // 4))
    if kind in {"slide", "rise"}:
        if left >= end - edge:
            a = (left - (end - edge)) / edge
            b = (right - (end - edge)) / edge
            return rf"\move({x},{y - round(distance * a)},{x},{y - round(distance * b)},0,{duration * 10})"
        a = min(1, (left - start) / edge)
        b = min(1, (right - start) / edge)
        return rf"\move({x},{y + round(distance * (1 - a))},{x},{y + round(distance * (1 - b))},0,{duration * 10})"
    # Drift is deliberately gentler than slide and travels horizontally.
    if left >= end - edge:
        a = (left - (end - edge)) / edge
        b = (right - (end - edge)) / edge
        return rf"\move({x + round(distance * a)},{y},{x + round(distance * b)},{y},0,{duration * 10})"
    a = min(1, (left - start) / edge)
    b = min(1, (right - start) / edge)
    return rf"\move({x - round(distance * (1 - a))},{y},{x - round(distance * (1 - b))},{y},0,{duration * 10})"


def _highlighted(words: list[_Word], split: int, selected: set[str], active: int | None,
                 visible: int | None, emphasize: bool, bold: bool, accent: str,
                 primary: str) -> str:
    parts = []
    for i, word in enumerate(words[:visible]):
        if i:
            parts.append(r"\N" if i == split else " ")
        highlight = i == active or (emphasize and _normalized(word.text) in selected)
        text = _escape(word.text)
        if highlight:
            reset = rf"\1c{primary}\b{1 if bold else 0}"
            text = "{" + rf"\1c{accent}\b1" + "}" + text + "{" + reset + "}"
        parts.append(text)
    return "".join(parts)


def _events(words, *, preset, mode, emphasize, selected, animation, x, y, size, available,
            accent, primary, width=1920, height=1080, anchor=2):
    groups = ([word] for word in words) if mode == "word" else _groups(words, size, available)
    for group in groups:
        split, fit = _layout(group, size, available)
        start = group[0].start
        end = min(max(word.end for word in group), start + 400)
        if end <= start:
            continue
        chosen = selected or set()
        if emphasize and selected is None:
            candidates = [word for word in group if len(_normalized(word.text)) >= 5
                          and _normalized(word.text) not in _STOPWORDS]
            chosen = {_normalized(word.text) for word in sorted(
                candidates, key=lambda word: -len(_normalized(word.text)))[:2]}
        boundaries = {start, end}
        karaoke = preset == "karaoke" and mode != "word"
        progressive = animation == "typewriter" and mode != "word"
        if karaoke:
            boundaries.update(time for word in group for time in (word.start, word.end)
                              if start < time < end)
        elif progressive:
            boundaries.update(word.start for word in group if start < word.start < end)
        # Split travel animations into entrance/hold and exit; each piece uses
        # a legal single-event \move.
        edge = min(15, (end - start) // 4)
        sliding = animation in {"slide", "rise", "drift"} and end - start >= 12
        if sliding:
            boundaries.add(end - edge)
        points = sorted(boundaries)
        for left, right in zip(points, points[1:]):
            active = next((i for i, word in enumerate(group)
                           if word.start <= left < word.end), None) if preset == "karaoke" else None
            visible = sum(word.start <= left for word in group) if progressive else None
            text = _highlighted(group, split, chosen, active, visible, emphasize,
                                bool(PRESETS[preset].get("bold", preset in {"punch", "pop", "outline"})),
                                accent, primary)
            # Use the whole phrase's animation clock on sliced events so a
            # changing active word does not retrigger the entrance animation.
            tags = _motion(animation, end - start, x, y, width=width, height=height)
            offset = (left - start) * 10
            length = (right - left) * 10
            if sliding:
                distance = max(1, round(fit * (0.18 if preset == "premiere" else 0.32)))
                if animation == "drift":
                    distance = max(1, round(distance * 0.55))
                tags = _travel_motion(animation, left, right, start, end, x, y, distance)
                if animation in {"slide", "rise"}:
                    # Entrance fades use absolute phrase time, including slices.
                    tags += rf"\fade(255,0,0,{-offset},{edge * 10 - offset},{length},{length})"
            elif len(points) > 2:
                # Retain a single continuous global animation on karaoke and
                # typewriter slices using shifted \t / \fade clocks.
                tags = _shift_motion(tags, offset, (end - start) * 10)
            prefix = "{" + rf"\an{anchor}\q2\fs{fit:.2f}" + tags + "}"
            yield left, right, prefix + text


def _shift_motion(tags: str, offset: int, total: int) -> str:
    tags = re.sub(r"\\t\((\d+),(\d+),", lambda m:
                  rf"\t({int(m[1]) - offset},{int(m[2]) - offset},", tags)
    return re.sub(r"\\fad\((\d+),(\d+)\)", lambda m:
                  rf"\fade(255,0,255,{-offset},{int(m[1]) - offset},"
                  rf"{total - int(m[2]) - offset},{total - offset})", tags)


def _position_values(position, width, height, vertical_safe_margin):
    aliases = {"lower": "bottom", "lower-third": "bottom", "upper": "top",
               "middle": "center"}
    position = aliases.get(str(position).strip().lower(), str(position).strip().lower())
    if position not in {"top", "center", "bottom"}:
        raise ValueError("position must be 'top', 'center', or 'bottom'")
    if vertical_safe_margin is None:
        margin = 0.20
    else:
        margin = float(vertical_safe_margin)
        if margin > 1:
            margin /= height
        if not math.isfinite(margin) or not 0 <= margin < 0.5:
            raise ValueError("vertical_safe_margin must be a finite fraction below 0.5")
    if position == "top":
        return round(width / 2), round(height * margin), 8, margin
    if position == "center":
        return round(width / 2), round(height * 0.5), 5, margin
    return round(width / 2), round(height * (1 - margin)), 2, margin


def _safe_ass_color(value, name):
    if isinstance(value, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", value):
        red, green, blue = value[1:3], value[3:5], value[5:7]
        value = f"&H00{blue}{green}{red}&"
    if isinstance(value, str) and re.fullmatch(r"#[0-9A-Fa-f]{8}", value):
        alpha, red, green, blue = value[1:3], value[3:5], value[5:7], value[7:9]
        value = f"&H{alpha}{blue}{green}{red}&"
    if not isinstance(value, str) or not re.fullmatch(r"&H[0-9A-Fa-f]{8}&", value):
        raise ValueError(f"{name} must be an ASS &HAABBGGRR& color")
    return value.upper()


def write_ass(segments, path, *, width=1920, height=1080, preset="clean",
              mode="line", emphasize=False, keywords=(), important_words=(),
              emphasis_words=(), animation="auto", font="Segoe UI", font_size=56,
              position="bottom", vertical_safe_margin=None, caption_position=None,
              safe_margin=None, panel_style=None, primary_color=None,
              accent_color=None, outline_color=None, background_color=None) -> Path:
    """Write UTF-8 ASS and return its Path; no engine or renderer is imported.

    ``font_size`` is measured at a frame width of 1080. By default captions are
    centered at 80% of frame height, constrained to 82% of width and at most two
    lines. ``position`` accepts top/center/bottom and ``vertical_safe_margin``
    accepts a normalized fraction (or pixels when above 1). ``mode='word'``
    displays only the actual word during its own interval.
    Animation overrides accept every name in :data:`ANIMATIONS`; auto uses the
    preset, including progressive typewriter builds. none disables motion.
    Explicit keywords and ``important_words`` match complete, case-insensitive
    normalized tokens.  The latter may be supplied by a transcription layer as
    already-normalized words.  ``important_words`` enables highlighting on its
    own; ``keywords`` retains the historical ``emphasize=True`` requirement.
    With emphasis enabled and no list, at most two long non-stopwords are
    accented.
    Empty input produces a valid ASS file with no dialogue. Unknown options,
    unsafe font names, and nonfinite dimensions/times raise ValueError.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown caption preset: {preset}")
    if mode not in {"line", "word"}:
        raise ValueError("mode must be 'line' or 'word'")
    if animation not in {"auto", *ANIMATIONS}:
        raise ValueError(f"Unknown animation: {animation}")
    values = (float(width), float(height), float(font_size))
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Frame dimensions and font_size must be positive and finite")
    width, height = round(values[0]), round(values[1])
    if width < 1 or height < 1:
        raise ValueError("Frame dimensions must round to at least one pixel")
    metadata = PRESETS[preset]
    if caption_position is not None:
        position = caption_position
    if safe_margin is not None:
        vertical_safe_margin = safe_margin
    x, y, anchor, margin_fraction = _position_values(position, width, height,
                                                      vertical_safe_margin)
    panel_style = metadata.get("panel_style", "outline") if panel_style is None else panel_style
    if panel_style not in {"none", "outline", "box", "solid", "accent"}:
        raise ValueError("panel_style must be one of none, outline, box, solid, accent")
    colors = {
        "primary_color": metadata["primary_color"] if primary_color is None else primary_color,
        "accent_color": metadata["accent_color"] if accent_color is None else accent_color,
        "outline_color": metadata["outline_color"] if outline_color is None else outline_color,
        "background_color": metadata["background_color"] if background_color is None else background_color,
    }
    colors = {key: _safe_ass_color(value, key) for key, value in colors.items()}
    if font is None or font == "auto":
        candidates = metadata["font_family_candidates"]
        font = candidates[0] if candidates else "Segoe UI"
    if not isinstance(font, str) or not font.strip() or any(
            char in font for char in ",\r\n{}\\") or any(ord(char) < 32 for char in font):
        raise ValueError("font must be a nonempty, single safe ASS field")
    if isinstance(keywords, str):
        keywords = (keywords,)
    if isinstance(important_words, str):
        important_words = (important_words,)
    if isinstance(emphasis_words, str):
        emphasis_words = (emphasis_words,)
    keywords = tuple(keywords)
    important_words = tuple(important_words)
    emphasis_words = tuple(emphasis_words)
    explicit = {
        _normalized(str(keyword))
        for keyword in (*keywords, *important_words, *emphasis_words)
        if _normalized(str(keyword))
    }
    # Preserve the old keyword/emphasize contract, while allowing a caller
    # that already has normalized important words to opt into highlighting
    # without needing a second boolean switch.
    highlight_requested = bool(emphasize or important_words or emphasis_words)
    # A supplied (even punctuation-only) list is an explicit request, so it
    # suppresses the historical heuristic instead of silently selecting other
    # words.
    selected = explicit if (keywords or important_words or emphasis_words) else None
    size = min(values[2] * width / 1080, height * 0.075)
    margin = round(width * 0.09)
    outline = size * float(metadata.get("outline_scale", 0.075 if preset == "outline" else 0.045))
    border = int(metadata.get("border", 3 if preset == "marker" else 1))
    if panel_style in {"box", "solid", "accent"}:
        border = 3
    background = colors["accent_color"] if panel_style == "accent" else colors["background_color"]
    shadow = size * 0.025
    # ASS spacing is expressed in script pixels; preset metadata uses small
    # readable values rather than a font-size multiplier.
    spacing = float(metadata.get("spacing", 0))
    bold = int(metadata.get("bold", -1 if preset in {"punch", "pop", "outline"} else 0))
    header = f"""[Script Info]
; Original generic caption treatment. Times are already on the output timeline.
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
; CaptionStyleId: {metadata['style_id']}; Panel: {panel_style}; SafeZone: {metadata['safe_zone']}; Easing: {metadata['easing']}
Style: Caption,{font.strip()},{size:.2f},{colors['primary_color']},{colors['accent_color']},{colors['outline_color']},{background},{bold},0,0,0,100,100,{spacing:.2f},0,{border},{outline:.2f},{shadow:.2f},{anchor},{margin},{margin},{height-y},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    motion = metadata["animation"] if animation == "auto" else animation
    events = []
    for segment in segments:
        events.extend(_events(_words(segment), preset=preset, mode=mode,
                              emphasize=highlight_requested, selected=selected, animation=motion,
                               x=x, y=y, size=size, available=width * 0.82,
                               accent=colors["accent_color"], primary=colors["primary_color"],
                               width=width, height=height, anchor=anchor))
    lines = [f"Dialogue: 0,{_stamp(start)},{_stamp(end)},Caption,,0,0,0,,{text}\n"
             for start, end, text in sorted(events, key=lambda item: (item[0], item[1]))]
    destination = Path(path)
    destination.write_text(header + "".join(lines), encoding="utf-8")
    return destination


def _preview_animation_tags(animation, x, y, width, height, distance):
    """Return deterministic entrance/exit tags for a UI preview model."""
    if animation in {"slide", "rise", "drift"}:
        entrance = _travel_motion(animation, 0, 85, 0, 100, x, y, distance)
        exit_tag = _travel_motion(animation, 85, 100, 0, 100, x, y, distance)
        return {"in": entrance, "hold": rf"\pos({x},{y})", "out": exit_tag}
    all_tags = _motion(animation, 100, x, y, width=width, height=height)
    return {"in": all_tags, "hold": rf"\pos({x},{y})", "out": all_tags}


def preview_model(text="Readable captions", *, width=1920, height=1080,
                  preset="clean", animation="auto", font_size=56,
                  position="bottom", vertical_safe_margin=None,
                  emphasis_words=(), important_words=()):
    """Build a frame-independent caption preview description.

    The result is plain data for a canvas or other UI: fitted line geometry,
    word emphasis flags, style metadata, and representative ASS tags for the
    entrance/hold/exit phases.  It deliberately has no frame clock, Tk import,
    font lookup, or renderer dependency, so every preset and animation can be
    previewed consistently by a caller.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown caption preset: {preset}")
    if animation not in {"auto", *ANIMATIONS}:
        raise ValueError(f"Unknown animation: {animation}")
    values = (float(width), float(height), float(font_size))
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Frame dimensions and font_size must be positive and finite")
    width, height = round(values[0]), round(values[1])
    if width < 1 or height < 1:
        raise ValueError("Frame dimensions must round to at least one pixel")
    x, y, anchor, margin_fraction = _position_values(position, width, height,
                                                      vertical_safe_margin)
    metadata = PRESETS[preset]
    motion = metadata["animation"] if animation == "auto" else animation
    size = min(values[2] * width / 1080, height * 0.075)
    tokens = _plain(text).split() or ["Caption"]
    words = [_Word(token, index * 10, (index + 1) * 10)
             for index, token in enumerate(tokens)]
    split, fit = _layout(words, size, width * 0.82)
    lines = [" ".join(word.text for word in words[:split]),
             " ".join(word.text for word in words[split:])]
    lines = [line for line in lines if line]
    line_widths = [round(_units(line) * fit, 2) for line in lines]
    line_height = round(fit * 1.2, 2)
    box_width = round(max(line_widths or [0]), 2)
    box_height = round(line_height * len(lines), 2)
    chosen = {_normalized(str(word)) for word in (*((important_words,) if isinstance(important_words, str) else important_words),
                                                   *((emphasis_words,) if isinstance(emphasis_words, str) else emphasis_words))
              if _normalized(str(word))}
    word_models = []
    cursor_by_line = [0.0, 0.0]
    for index, word in enumerate(words):
        line_index = 0 if index < split else 1
        if line_index >= len(lines):
            line_index = len(lines) - 1
        if index and index != split:
            cursor_by_line[line_index] += _units(" ") * fit
        word_width = round(_units(word.text) * fit, 2)
        word_models.append({
            "index": index,
            "text": word.text,
            "line": line_index,
            "x": round(x - box_width / 2 + cursor_by_line[line_index], 2),
            "y": round(y - box_height / 2 + line_index * line_height, 2),
            "width": word_width,
            "height": line_height,
            "emphasized": _normalized(word.text) in chosen,
            "tag": rf"\1c{metadata['accent_color']}\b1" if _normalized(word.text) in chosen else "",
        })
        cursor_by_line[line_index] += word_width
    tags = _preview_animation_tags(motion, x, y, width, height, max(1, round(fit * .32)))
    style = {key: value for key, value in metadata.items()
             if key in {"style_id", "label", "description", "panel_style", "panel_opacity",
                        "safe_zone", "easing", "animation_in", "animation_out",
                        "emphasis_style", "primary_hex", "accent_hex", "outline_hex",
                        "background_hex"}}
    return {
        "version": 1,
        "preset": preset,
        "animation": motion,
        "available_presets": tuple(PRESETS),
        "available_animations": ANIMATIONS,
        "style": style,
        "geometry": {
            "frame": {"width": width, "height": height},
            "anchor": anchor,
            "x": x,
            "y": y,
            "safe_margin": margin_fraction,
            "line_count": len(lines),
            "lines": tuple(lines),
            "font_size": round(fit, 2),
            "width": box_width,
            "height": box_height,
        },
        "words": tuple(word_models),
        "tags": tags,
    }


# Friendly aliases make the model discoverable to small integrations without
# forcing them to know whether the project calls it a caption or preview model.
caption_preview = preview_model
preview_caption = preview_model


__all__ = ["ANIMATION_METADATA", "ANIMATIONS", "PRESETS", "caption_preview",
           "preview_caption", "preview_model", "write_ass"]
