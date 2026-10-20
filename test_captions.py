"""Behavior checks for the standalone ASS writer (standard library only)."""

import math
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    from .captions import ANIMATIONS, ANIMATION_METADATA, PRESETS, preview_model, write_ass
except ImportError:
    from captions import ANIMATIONS, ANIMATION_METADATA, PRESETS, preview_model, write_ass


def word(text, start, end):
    return SimpleNamespace(text=text, start=start, end=end)


def segment(text="", start=0, end=3, words=()):
    return SimpleNamespace(text=text, start=start, end=end, words=words)


def ticks(stamp):
    hours, minutes, seconds = stamp.split(":")
    return round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 100)


def dialogues(ass):
    result = []
    for line in ass.splitlines():
        if line.startswith("Dialogue:"):
            fields = line.split(",", 9)
            result.append((ticks(fields[1]), ticks(fields[2]), fields[9]))
    return result


def plain(text):
    return re.sub(r"\{[^}]*\}", "", text).replace(r"\N", " ")


class CaptionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "字幕.ass"

    def render(self, segments, **options):
        result = write_ass(segments, self.path, **options)
        self.assertEqual(result, self.path)
        return self.path.read_text(encoding="utf-8")

    def test_presets_and_metadata(self):
        requested = {
            "alex_hormozi", "mrbeast", "karaoke", "iman_gadzhi", "ali_abdaal",
            "devin_jatho", "luxury_editorial", "tiktok_native_box", "modern_podcast",
            "documentary_cinematic", "news_explainer", "dark_motivational", "neon_gaming",
            "retro_vhs", "handwritten_vlog",
        }
        legacy = {"clean", "punch", "karaoke", "marker", "pop", "slide",
                  "typewriter", "blur", "premiere", "outline"}
        self.assertTrue(requested.issubset(PRESETS))
        self.assertTrue(legacy.issubset(PRESETS))
        self.assertEqual(len(requested - {"karaoke"}), 14)
        for name, preset in PRESETS.items():
            with self.subTest(preset=name):
                for key in ("label", "description", "animation", "font_family_candidates",
                            "primary_color", "accent_color", "outline_color",
                            "background_color", "color_grade_guidance"):
                    self.assertTrue(preset[key])
                self.assertTrue(preset["font_family_candidates"])
                self.assertIn(preset["animation"], {"none", "fade", "pop", "slide", "blur", "typewriter"})
        self.assertEqual(PRESETS["premiere"]["label"], "Premiere-inspired clean")

    def test_new_preset_metadata_drives_style_and_auto_animation(self):
        data = [segment(words=[word("important", 0, 0.6), word("word", 0.6, 1.2)])]
        for name in {
            "alex_hormozi", "mrbeast", "iman_gadzhi", "ali_abdaal", "devin_jatho",
            "luxury_editorial", "tiktok_native_box", "modern_podcast",
            "documentary_cinematic", "news_explainer", "dark_motivational",
            "neon_gaming", "retro_vhs", "handwritten_vlog",
        }:
            with self.subTest(preset=name):
                ass = self.render(data, preset=name, important_words=["important"])
                self.assertIn(PRESETS[name]["primary_color"], ass)
                self.assertIn(PRESETS[name]["accent_color"], ass)
                self.assertIn(r"\1c" + PRESETS[name]["accent_color"] + r"\b1}important", ass)
                if PRESETS[name]["animation"] == "pop":
                    self.assertIn(r"\fscx88", ass)

    def test_important_words_are_normalized_and_use_preset_accent(self):
        data = [segment(words=[word("Café!", 0, 0.4), word("ordinary", 0.4, 0.8)])]
        ass = self.render(data, preset="luxury_editorial", important_words=["CAFE\u0301"])
        text = dialogues(ass)[0][2]
        self.assertIn(r"\1c" + PRESETS["luxury_editorial"]["accent_color"] + r"\b1}Café!", text)
        self.assertNotIn(r"\b1}ordinary", text)

    def test_auto_font_uses_preset_family_candidates(self):
        ass = self.render([segment("font candidate")], preset="handwritten_vlog", font=None)
        self.assertIn("Caption," + PRESETS["handwritten_vlog"]["font_family_candidates"][0] + ",", ass)

    def test_legacy_keys_still_write_ass(self):
        for preset in ("clean", "punch", "karaoke", "marker", "pop", "slide",
                       "typewriter", "blur", "premiere", "outline"):
            with self.subTest(preset=preset):
                self.assertTrue(dialogues(self.render([segment("legacy caption")], preset=preset)))

    def test_word_mode_uses_output_timeline_exactly(self):
        source = segment("old source text", 100, 110, [
            word("Hello", 0.23, 0.64), word("世界", 0.91, 1.38)])
        entries = dialogues(self.render([source], mode="word", animation="none"))
        self.assertEqual([(a, b, plain(c)) for a, b, c in entries],
                         [(23, 64, "Hello"), (91, 138, "世界")])

    def test_phrases_are_rebuilt_from_kept_words(self):
        source = segment("um like keep this useful idea", words=[
            word("keep", 0.1, 0.4), word("this", 0.4, 0.7),
            word("useful", 0.7, 1.1), word("idea", 1.1, 1.5)])
        for preset in ("clean", "karaoke", "typewriter"):
            with self.subTest(preset=preset):
                entries = dialogues(self.render([source], preset=preset))
                content = " ".join(plain(c) for _, _, c in entries)
                self.assertNotIn("um", content.split())
                self.assertNotIn("like", content.split())
                self.assertIn("idea", content)

    def test_like_is_not_mandatorily_removed(self):
        ass = self.render([segment(words=[word("like", 0, 1)])], animation="none")
        self.assertEqual(plain(dialogues(ass)[0][2]), "like")

    def test_phrase_groups_have_at_most_two_short_lines(self):
        words = [word(f"word{i}", i * 0.25, (i + 1) * 0.25) for i in range(25)]
        entries = dialogues(self.render([segment(words=words)], animation="none"))
        self.assertGreaterEqual(len(entries), 3)
        rebuilt = []
        for _, _, text in entries:
            content = re.sub(r"\{[^}]*\}", "", text)
            lines = content.split(r"\N")
            self.assertLessEqual(len(lines), 2)
            for line in lines:
                self.assertLessEqual(len(line.split()), 6)
                rebuilt.extend(line.split())
        self.assertEqual(rebuilt, [w.text for w in words])

    def test_keyword_case_punctuation_and_unicode_normalization(self):
        words = [word("Brilliant!", 0, 0.4), word("Café", 0.4, 0.9),
                 word("ordinary", 0.9, 1.3)]
        text = dialogues(self.render([segment(words=words)], emphasize=True,
                                    keywords=["BRILLIANT", "CAFE\u0301"],
                                    animation="none"))[0][2]
        self.assertRegex(text, r"\\1c&H00BCDFA0&\\b1\}Brilliant!")
        self.assertRegex(text, r"\\1c&H00BCDFA0&\\b1\}Café")
        self.assertNotRegex(text, r"\\b1\}ordinary")

    def test_keywords_require_emphasis_and_do_not_select_unlisted_words(self):
        data = [segment(words=[word("wonderful", 0, 0.4), word("different", 0.4, 1)])]
        text = dialogues(self.render(data, emphasize=False, keywords=["wonderful"]))[0][2]
        self.assertNotIn(r"\1c", text)
        text = dialogues(self.render(data, emphasize=True, keywords=["absent"]))[0][2]
        self.assertNotIn(r"\1c", text)
        text = dialogues(self.render(data, emphasize=True, keywords=["!!!"]))[0][2]
        self.assertNotIn(r"\1c", text)

    def test_uneven_word_lengths_do_not_create_overlong_lines(self):
        data = [segment(words=[word(text, i * 0.2, (i + 1) * 0.2)
                               for i, text in enumerate(["extraordinary"] + ["a"] * 9)])]
        for _, _, text in dialogues(self.render(data, animation="none")):
            content = re.sub(r"\{[^}]*\}", "", text)
            self.assertTrue(all(len(line.split()) <= 6 for line in content.split(r"\N")))

    def test_heuristic_is_conservative_and_bounded(self):
        words = [word(text, i * 0.2, (i + 1) * 0.2) for i, text in enumerate(
            ["there", "should", "creative", "remarkable", "beautiful"])]
        text = dialogues(self.render([segment(words=words)], emphasize=True))[0][2]
        self.assertEqual(text.count(r"\1c&H00BCDFA0&"), 2)
        self.assertNotRegex(text, r"\\b1\}(there|should)")

    def test_karaoke_highlight_tracks_word_times_and_gaps(self):
        data = [segment(words=[word("one", 1, 1.25), word("two", 1.50, 2)])]
        entries = dialogues(self.render(data, preset="karaoke"))
        self.assertEqual([(a, b) for a, b, _ in entries],
                         [(100, 125), (125, 150), (150, 200)])
        self.assertIn(r"\b1}one", entries[0][2])
        self.assertNotIn(r"\b1", entries[1][2])
        self.assertIn(r"\b1}two", entries[2][2])
        self.assertTrue(all(plain(text) == "one two" for _, _, text in entries))

    def test_typewriter_builds_on_word_boundaries(self):
        data = [segment(words=[word("one", 1, 1.2), word("two", 1.4, 1.6),
                               word("three", 1.8, 2)])]
        entries = dialogues(self.render(data, preset="typewriter"))
        self.assertEqual([(a, b, plain(c)) for a, b, c in entries],
                         [(100, 140, "one"), (140, 180, "one two"),
                          (180, 200, "one two three")])
        disabled = dialogues(self.render(data, preset="typewriter", animation="none"))
        self.assertEqual([(a, b, plain(c)) for a, b, c in disabled],
                         [(100, 200, "one two three")])

    def test_word_mode_karaoke_is_one_active_word(self):
        entries = dialogues(self.render([segment(words=[word("hello", 1, 1.5)])],
                                        preset="karaoke", mode="word"))
        self.assertEqual([(a, b, plain(c)) for a, b, c in entries], [(100, 150, "hello")])
        self.assertIn(r"\b1}hello", entries[0][2])

    def test_short_durations_round_to_visible_tick_without_motion(self):
        for animation in ("fade", "pop", "slide", "blur"):
            with self.subTest(animation=animation):
                ass = self.render([segment(words=[word("I", 1.004, 1.006),
                                                  word("a", 1.02, 1.02001)])],
                                  mode="word", animation=animation)
                entries = dialogues(ass)
                self.assertEqual([(a, b) for a, b, _ in entries], [(100, 101), (102, 103)])
                for _, _, text in entries:
                    self.assertNotRegex(text, r"\\(fad|fade|t|move|blur|fsc)")

    def test_timestamp_carries_and_duration_bounds(self):
        ass = self.render([segment(words=[word("carry", 3599.999, 3600.035),
                                          word("hold", 3601, 3700)])],
                          mode="word", animation="none")
        self.assertIn("1:00:00.00,1:00:00.04", ass)
        self.assertTrue(all(0 < b - a <= 400 for a, b, _ in dialogues(ass)))

    def test_negative_zero_reversed_and_nonfinite_times(self):
        ass = self.render([segment(words=[word("negative", -1, 0.2),
                                          word("zero", 1, 1), word("reverse", 2, 1)])])
        entries = dialogues(ass)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0][0], 0)
        self.assertEqual(plain(entries[0][2]), "negative")
        for bad in (math.inf, -math.inf, math.nan):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.render([segment(words=[word("bad", bad, 3)])])

    def test_no_fallback_when_supplied_words_are_invalid(self):
        ass = self.render([segment("removed filler", words=[word("invalid", 1, 0)])])
        self.assertEqual(dialogues(ass), [])

    def test_escaping_blocks_ass_and_event_injection(self):
        payload = "{\\pos(0,0)}\\N\\n\nDialogue: 0,{\\p1} café 世界"
        ass = self.render([segment(words=[word(payload, 0, 2)])], animation="none")
        texts = [text for _, _, text in dialogues(ass)]
        content = " ".join(plain(text) for text in texts)
        self.assertIn("｛＼pos(0,0)｝＼N＼n", content)
        self.assertIn("café 世界", content)
        self.assertNotIn(r"\pos(0,0)", ass)
        self.assertNotIn(r"\p1", ass)
        self.assertTrue(all(text.count("{") == text.count("}") for text in texts))

    def test_empty_and_fallback_are_deterministic_unicode(self):
        ass = self.render([])
        self.assertIn("[Events]", ass)
        self.assertEqual(dialogues(ass), [])
        source = [segment("héllo 世界 again", 2, 3.2)]
        first = self.render(source, mode="word", animation="none")
        self.assertEqual(self.render(source, mode="word", animation="none"), first)
        self.assertEqual([(a, b, plain(c)) for a, b, c in dialogues(first)],
                         [(200, 240, "héllo"), (240, 280, "世界"), (280, 320, "again")])

    def test_vertical_size_safe_position_and_width_scaling(self):
        source = [segment("readable captions")]
        portrait = self.render(source, width=1080, height=1920)
        self.assertIn("PlayResX: 1080", portrait)
        self.assertIn("PlayResY: 1920", portrait)
        self.assertIn("Caption,Segoe UI,56.00,", portrait)
        self.assertIn(r"\pos(540,1536)", portrait)
        half = self.render(source, width=540, height=960)
        self.assertIn("Caption,Segoe UI,28.00,", half)
        self.assertIn(r"\pos(270,768)", half)
        landscape = self.render(source, width=1920, height=1080)
        self.assertIn("Caption,Segoe UI,81.00,", landscape)
        self.assertIn(r"\pos(960,864)", landscape)

    def test_long_unicode_token_is_fitted(self):
        ass = self.render([segment(words=[word("漢" * 40, 0, 1)])],
                          width=1080, height=1920, mode="word")
        text = dialogues(ass)[0][2]
        fit = float(re.search(r"\\fs([\d.]+)", text)[1])
        self.assertLess(fit, 56)
        self.assertLessEqual(40 * 1.05 * fit, 1080 * 0.82 + 0.5)
        self.assertEqual(plain(text), "漢" * 40)

    def test_animation_strings_and_explicit_overrides(self):
        data = [segment(words=[word("Motion", 0, 2)])]
        cases = {
            "none": (r"\pos",),
            "fade": (r"\fad(",),
            "pop": (r"\fscx88", r"\t(0,150,\fscx100", r"\t(1850,2000,\fscx88"),
            "slide": (r"\move(", r"\fade("),
            "blur": (r"\blur4", r"\t(0,150,\blur0)", r"\t(1850,2000,\blur4)"),
        }
        for motion, tags in cases.items():
            with self.subTest(motion=motion):
                ass = self.render(data, preset="pop", animation=motion)
                for tag in tags:
                    self.assertIn(tag, ass)
                if motion == "none":
                    self.assertNotIn(r"\t(", ass)
                    self.assertNotIn(r"\fad", ass)
        slide = dialogues(self.render(data, animation="slide"))
        self.assertEqual([(a, b) for a, b, _ in slide], [(0, 185), (185, 200)])
        self.assertIn(r"\move(960,890,960,864", slide[0][2])
        self.assertIn(r"\move(960,864,960,838", slide[1][2])

    def test_advanced_animation_catalog_has_in_out_timing(self):
        self.assertTrue({"wipe", "spring", "rise", "drift"}.issubset(ANIMATIONS))
        for animation in ANIMATIONS:
            with self.subTest(animation=animation):
                metadata = ANIMATION_METADATA[animation]
                self.assertIn("easing", metadata)
                self.assertGreaterEqual(metadata["in_ms"], 0)
                self.assertGreaterEqual(metadata["out_ms"], 0)
        source = [segment(words=[word("Motion", 0, 2)])]
        markers = {
            "wipe": r"\clip(", "spring": r"\fscx82", "rise": r"\move(",
            "drift": r"\move(",
        }
        for animation, marker in markers.items():
            with self.subTest(animation=animation):
                ass = self.render(source, animation=animation)
                self.assertIn(marker, ass)

    def test_position_safe_margin_panel_and_per_word_emphasis_options(self):
        source = [segment(words=[word("Top", 0, 1), word("signal", 1, 2)])]
        ass = self.render(source, position="top", vertical_safe_margin=0.1,
                          panel_style="solid", emphasis_words=["signal"])
        self.assertIn(r"\an8\q2", ass)
        self.assertIn("Style: Caption,Segoe UI,", ass)
        self.assertIn("Panel: solid", ass)
        self.assertIn(r"\1c" + PRESETS["clean"]["accent_color"] + r"\b1}signal", ass)

    def test_preview_model_is_deterministic_and_covers_catalog(self):
        for preset in PRESETS:
            for animation in ANIMATIONS:
                with self.subTest(preset=preset, animation=animation):
                    model = preview_model("A useful preview", preset=preset,
                                          animation=animation, position="center")
                    self.assertEqual(model, preview_model("A useful preview", preset=preset,
                                                          animation=animation, position="center"))
                    self.assertEqual(model["animation"], animation)
                    self.assertEqual(model["geometry"]["anchor"], 5)
                    self.assertTrue(model["geometry"]["lines"])
                    self.assertTrue(model["tags"]["in"])
                    self.assertTrue(model["tags"]["out"])
        model = preview_model("Keep this", emphasis_words=["keep"])
        self.assertTrue(model["words"][0]["emphasized"])
        self.assertIn("panel_style", model["style"])

    def test_all_presets_produce_readable_dialogue(self):
        for preset in PRESETS:
            with self.subTest(preset=preset):
                entries = dialogues(self.render([segment("A useful original caption")], preset=preset))
                self.assertTrue(entries)
                self.assertTrue(all(end > start and plain(text) for start, end, text in entries))
                self.assertIn("caption", plain(entries[-1][2]))

    def test_invalid_options_and_font_header_injection(self):
        for options in ({"preset": "missing"}, {"mode": "paragraph"},
                        {"animation": "spin"}, {"width": 0}, {"height": -1},
                        {"width": 0.1}, {"width": math.inf}, {"font_size": math.nan},
                        {"font": "Bad,Font"}, {"font": "Font\n[Events]"},
                        {"font": "{\\pos(0,0)}"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.render([], **options)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_ffmpeg_libass_render_smoke(self):
        ffmpeg = shutil.which("ffmpeg")
        filters = subprocess.run([ffmpeg, "-hide_banner", "-filters"],
                                 capture_output=True, timeout=30)
        if not re.search(rb"\bass\s+V->V", filters.stdout):
            self.skipTest("This FFmpeg build has no libass filter")
        # Verify real rendered pixels, not only ASS syntax. No video artifacts
        # are retained; raw grayscale frames travel through the subprocess pipe.
        for preset in PRESETS:
            with self.subTest(preset=preset):
                self.render([segment(words=[word("Readable", 0, 0.1),
                                            word("mint", 0.1, 0.6)])],
                            preset=preset, width=360, height=640)
                process = subprocess.run([
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-filter_threads", "1",
                    "-f", "lavfi", "-i", "color=c=black:s=360x640:r=10:d=1",
                    "-vf", f"ass=filename={self.path.name}", "-frames:v", "3",
                    "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1",
                ], cwd=self.path.parent, capture_output=True, timeout=30)
                self.assertEqual(process.returncode, 0, process.stderr.decode("utf-8", "replace"))
                self.assertEqual(len(process.stdout), 360 * 640 * 3)
                third_frame = process.stdout[360 * 640 * 2:]
                self.assertGreater(max(third_frame), 100, "No bright caption pixels rendered")


if __name__ == "__main__":
    unittest.main()
