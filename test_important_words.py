"""Behavior checks for the dependency-free important-word scorer."""

import unittest
from types import SimpleNamespace

try:
    from .important_words import score_important_words
except ImportError:
    from important_words import score_important_words


def word(text):
    return SimpleNamespace(text=text, start=0, end=1)


class ImportantWordsTests(unittest.TestCase):
    def test_filters_fillers_and_stopwords(self):
        words = [word(text) for text in ("um", "the", "uh", "and", "camera", "editing")]
        self.assertEqual(score_important_words(words), ("camera", "editing"))

    def test_explicit_terms_are_first_and_normalized(self):
        words = [word(text) for text in ("camera", "timeline", "rendering")]
        self.assertEqual(
            score_important_words(words, explicit=("TIMELINE!", "Missing")),
            ("timeline", "missing", "rendering"),
        )

    def test_output_is_deterministic_and_first_ordered(self):
        words = [word(text) for text in ("short", "API", "transcript", "42", "short")]
        expected = ("api", "transcript", "42")
        self.assertEqual(score_important_words(words), expected)
        self.assertEqual(score_important_words(words), expected)

    def test_case_and_punctuation_are_normalized(self):
        words = [word(text) for text in ("Café!", "CAFE\u0301", "build,", "v2.")]
        self.assertEqual(score_important_words(words, max_words=4), ("café", "build", "v2"))

    def test_max_words_caps_explicit_and_heuristic_terms(self):
        words = [word(text) for text in ("alpha", "bravo", "charlie", "delta")]
        self.assertEqual(score_important_words(words, max_words=2), ("alpha", "charlie"))
        self.assertEqual(score_important_words(words, max_words=2, explicit=("alpha", "bravo", "echo")),
                         ("alpha", "bravo"))

    def test_empty_input(self):
        self.assertEqual(score_important_words(()), ())
        self.assertEqual(score_important_words((), explicit=("!!!",)), ())
        self.assertEqual(score_important_words((), max_words=0, explicit=("keyword",)), ())


if __name__ == "__main__":
    unittest.main()
