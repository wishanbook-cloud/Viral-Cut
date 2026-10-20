"""Focused tests for the local four-feature media pipeline."""

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from engine import (
    Cut,
    DependencyError,
    EditorConfig,
    FFmpegError,
    TranscriptSegment,
    TranscriptWord,
    _align_cuts,
    _retime_transcript,
    apply_face_tracking,
    build_cuts,
    filler_word_ranges,
    process_video,
    probe_media,
    render_cuts,
)


class CoreUnitTests(unittest.TestCase):
    def test_filler_ranges_and_caption_retiming_stay_synchronized(self):
        segment = TranscriptSegment(
            "Hello um world",
            0,
            3,
            (TranscriptWord("Hello", 0, 0.8), TranscriptWord("um", 1, 1.5), TranscriptWord("world", 2, 3)),
        )
        fillers = filler_word_ranges(segment.words)
        cuts = build_cuts(3, filler_ranges=fillers)
        retimed = _retime_transcript((segment,), cuts)
        self.assertEqual([word.text for item in retimed for word in item.words], ["Hello", "world"])
        self.assertNotIn("um", " ".join(item.text for item in retimed))
        self.assertAlmostEqual(retimed[-1].words[0].start, 1.5)

    def test_cut_planning_merges_silence_and_filler_ranges(self):
        cuts = build_cuts(5, silence_ranges=[(1, 2), (1.8, 2.5)], filler_ranges=[(3, 3.5)])
        self.assertEqual([(cut.source_start, cut.source_end) for cut in cuts], [(0, 1), (2.5, 3), (3.5, 5)])

    def test_frame_alignment_keeps_timeline_continuous(self):
        cuts = _align_cuts([Cut(0.012, 0.989, 0), Cut(1.008, 2.069, 0.977)])
        self.assertAlmostEqual(cuts[1].timeline_start, cuts[0].duration)
        for cut in cuts:
            self.assertAlmostEqual(cut.duration * 30, round(cut.duration * 30))

    def test_dependency_error_is_clear_for_missing_whisper(self):
        with patch.dict("sys.modules", {"faster_whisper": None}):
            from engine import download_whisper_model

            with self.assertRaisesRegex(DependencyError, "faster-whisper"):
                download_whisper_model("tiny", "cpu", "int8")

    def test_input_and_output_contract(self):
        with self.assertRaises(ValueError):
            process_video("input.mp4", "input.mp4")
        with self.assertRaises(ValueError):
            process_video("input.mp4", "output.mov")


class ProcessFlowTests(unittest.TestCase):
    def _run_mocked(self, options, *, media=None, transcript=()):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "source.mp4"
            destination = folder / "output.mp4"
            source.write_bytes(b"source")
            media = media or SimpleNamespace(width=320, height=180, has_audio=True, duration=1.0, path=source)
            edit_result = SimpleNamespace(
                silence_ranges=(), filler_ranges=(), cuts=(), media=media,
            )

            def fake_edit(_source, output, **_kwargs):
                output.write_bytes(b"rendered")
                return edit_result

            with patch("engine.probe_media", return_value=media), \
                    patch("engine.VideoEditor") as editor_class, \
                    patch("engine.transcribe_media", return_value=(transcript, tuple(word for item in transcript for word in item.words))) as transcribe, \
                    patch("engine.apply_face_tracking") as tracker, \
                    patch("engine.apply_media_overlays") as overlays, \
                    patch("engine.analyze_transcript_flow") as flow, \
                    patch("engine.reframe_video") as reframe:
                editor_class.return_value.edit.side_effect = fake_edit
                def fake_tracking(_source, output, **_kwargs):
                    output.write_bytes(b"tracked")
                    return output

                tracker.side_effect = fake_tracking
                result = process_video(
                    str(source),
                    str(destination),
                    options,
                )

            return result, destination.is_file(), transcribe, tracker, overlays, flow, reframe, editor_class

    def test_active_flow_bypasses_legacy_optional_features(self):
        result, destination_exists, transcribe, tracker, overlays, flow, reframe, editor = self._run_mocked({
            "remove_silence": False,
            "remove_filler_words": False,
            "face_tracking": True,
            "captions": False,
            "analyze_flow": True,
            "remove_no_dialogue": True,
            "watermark_path": "missing.png",
            "music_path": "missing.wav",
            "transition": "mix",
            "face_tracking_mode": "Reaction split · Side-by-side",
            "output_format": "vertical",
        })
        self.assertTrue(destination_exists)
        self.assertFalse(transcribe.called)
        self.assertFalse(overlays.called)
        self.assertFalse(flow.called)
        self.assertFalse(reframe.called)
        self.assertEqual(tracker.call_args.kwargs["tracking_mode"], "single_subject")
        self.assertEqual(tracker.call_args.kwargs["output_format"], "original")
        self.assertEqual(editor.return_value.edit.call_args.kwargs["extra_removals"], ())
        self.assertEqual(result["transition"], "none")
        self.assertFalse(result["music"])
        self.assertFalse(result["watermark"])

    def test_whisper_is_only_started_for_fillers_or_captions_and_is_cpu_only(self):
        transcript = (TranscriptSegment("Hello", 0, 1, (TranscriptWord("Hello", 0, 1),)),)
        result, _destination_exists, transcribe, _tracker, _overlays, _flow, _reframe, _editor = self._run_mocked({
            "remove_silence": False,
            "remove_filler_words": True,
        }, transcript=transcript)
        self.assertTrue(transcribe.called)
        self.assertEqual(transcribe.call_args.kwargs["device"], "cpu")
        self.assertEqual(transcribe.call_args.kwargs["compute_type"], "int8")
        self.assertEqual(result["removed_filler_ranges"], 0)

    def test_speech_features_require_audio(self):
        media = SimpleNamespace(width=320, height=180, has_audio=False, duration=1.0, path=Path("source.mp4"))
        with self.assertRaisesRegex(FFmpegError, "require an audio stream"):
            self._run_mocked({"captions": True}, media=media)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class MediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.source = cls.folder / "source.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            "testsrc2=size=320x180:rate=30:duration=3", "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=48000:duration=3", "-c:v", "libx264",
            "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(cls.source),
        ], check=True, capture_output=True)
        cls.info = probe_media(cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_ffmpeg_cut_render_preserves_audio(self):
        output = self.folder / "cuts.mp4"
        render_cuts(
            self.source,
            output,
            self.info,
            _align_cuts([Cut(0, 1, 0), Cut(2, 3, 1)]),
            config=EditorConfig(preset="ultrafast", transition="mix"),
        )
        info = probe_media(output)
        self.assertTrue(info.has_audio)
        self.assertAlmostEqual(info.duration, 2, delta=0.08)

    def test_caption_pipeline_burns_subtitles_and_preserves_audio(self):
        output = self.folder / "captioned.mp4"
        transcript = (TranscriptSegment("Hello world", 0.2, 1.2, (
            TranscriptWord("Hello", 0.2, 0.7), TranscriptWord("world", 0.75, 1.2),
        )),)
        with patch("engine.transcribe_media", return_value=(transcript, transcript[0].words)):
            result = process_video(str(self.source), str(output), {
                "remove_silence": False,
                "captions": True,
            })
        info = probe_media(output)
        self.assertTrue(info.has_audio)
        self.assertTrue(result["captions"])
        self.assertTrue(output.with_suffix(".srt").is_file())
        self.assertTrue(output.with_suffix(".ass").is_file())

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required")
    def test_face_tracking_preserves_audio(self):
        output = self.folder / "tracked.mp4"
        apply_face_tracking(self.source, output)
        info = probe_media(output)
        self.assertTrue(info.has_audio)
        self.assertEqual((info.width, info.height), (320, 180))


if __name__ == "__main__":
    unittest.main()
