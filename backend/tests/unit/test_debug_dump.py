"""
Debug face dumps (app/core/debug_dump.py): off by default, one folder per
pipeline run when enabled, and never able to break the pipeline.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest

from app.core import debug_dump
from app.core.debug_dump import dump_detection, dump_note, dump_stage, face_dump
from app.core.inference_executor import run_inference

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 16


@pytest.fixture
def enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(debug_dump.settings, "debug_face_dump", True)
    monkeypatch.setattr(debug_dump.settings, "debug_face_dump_dir", str(tmp_path / "dumps"))
    return tmp_path / "dumps"


def _manifest(run_dir):
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _face(value=128):
    return np.full((40, 40, 3), value, np.uint8)


class TestDisabled:
    def test_is_a_no_op_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(debug_dump.settings, "debug_face_dump_dir", str(tmp_path / "dumps"))
        assert debug_dump.settings.debug_face_dump is False

        with face_dump("search", b"\xff\xd8\xff jpeg") as dump:
            dump_stage("v4_02_aligned_112", _face())
            dump_detection("v4_01_detect", _face(), (0, 0, 10, 10))
            dump_note("result", {"status": "MATCH"})

        assert dump is None
        assert not (tmp_path / "dumps").exists()


class TestEnabled:
    def test_writes_input_stages_and_manifest(self, enabled):
        jpeg = b"\xff\xd8\xff" + b"jpeg bytes"
        with face_dump("register", jpeg) as dump:
            dump_stage("v4_02_aligned_112", _face(), score=0.9)
            dump_stage("v3_09_final_112", np.full((20, 20, 3), 0.5, np.float32), rgb=True)
            dump_stage("gray", np.zeros((20, 20), np.uint8))
            dump_note("result", {"status": "REGISTERED", "face_id": 6})

        run_dir = dump.dir
        assert run_dir.parent == enabled and run_dir.name.split("_")[1] == "register"
        assert (run_dir / "00_input.jpg").read_bytes() == jpeg       # upload kept byte for byte
        manifest = _manifest(run_dir)
        assert manifest["flow"] == "register"
        assert manifest["result"] == {"status": "REGISTERED", "face_id": 6}
        assert manifest["settings"]["embedding_version"] == debug_dump.settings.embedding_version
        files = [s["file"] for s in manifest["stages"]]
        assert files == sorted(files)
        stage = next(s for s in manifest["stages"] if s["file"] == "v4_02_aligned_112.png")
        assert stage["score"] == 0.9 and stage["shape"] == [40, 40, 3]
        assert stage["lum_mean"] == 128.0 and stage["pct_dark"] == 0.0
        final = next(s for s in manifest["stages"] if s["file"] == "v3_09_final_112.png")
        assert final["lum_mean"] == pytest.approx(127, abs=1)    # float [0, 1] crop scaled to uint8

    def test_detection_draws_box_and_records_face_region(self, enabled):
        frame = np.zeros((100, 100, 3), np.uint8)
        frame[20:60, 30:70] = 200
        landmarks = np.array([[40, 35], [60, 35], [50, 45], [42, 52], [58, 52]], np.float32)

        with face_dump("search") as dump:
            dump_detection("v4_01_detect", frame, (30, 20, 70, 60), landmarks, label="scrfd 0.91", score=0.91)

        stage = _manifest(dump.dir)["stages"][0]
        assert stage["bbox"] == [30, 20, 70, 60] and stage["score"] == 0.91
        assert stage["face_region"]["lum_mean"] == 200.0          # stats of the box, not the frame
        assert not (dump.dir / "00_input.img").exists()            # no upload passed

    def test_manifest_is_written_when_the_pipeline_raises(self, enabled):
        with pytest.raises(RuntimeError):
            with face_dump("search", b"raw") as dump:
                dump_note("stage", "detect")
                raise RuntimeError("engine failed")

        manifest = _manifest(dump.dir)
        assert manifest["stage"] == "detect"
        assert (dump.dir / "00_input.img").read_bytes() == b"raw"

    def test_unwritable_dump_dir_disables_the_run(self, enabled):
        enabled.parent.mkdir(parents=True, exist_ok=True)
        enabled.write_text("a file, not a directory")
        with face_dump("search", b"raw") as dump:
            dump_stage("v4_02_aligned_112", _face())   # no active run: ignored
        assert dump is None

    def test_failed_stage_write_never_reaches_the_pipeline(self, enabled):
        with face_dump("search") as dump, patch("cv2.imwrite", side_effect=OSError("disk full")):
            dump_stage("v4_02_aligned_112", _face())
            dump_detection("v4_01_detect", _face(), (0, 0, 10, 10))
        assert _manifest(dump.dir)["stages"] == []

    async def test_stages_on_the_inference_pool_reach_the_run(self, enabled):
        with face_dump("search") as dump:
            await run_inference(dump_stage, "v4_02_aligned_112", _face())
        assert (dump.dir / "v4_02_aligned_112.png").exists()


class TestHelpers:
    @pytest.mark.parametrize("data, ext", [
        (b"\xff\xd8\xff\xe0", "jpg"),
        (PNG, "png"),
        (b"RIFF\0\0\0\0WEBPVP8 ", "webp"),
        (b"GIF89a", "img"),
    ])
    def test_guess_ext(self, data, ext):
        assert debug_dump._guess_ext(data) == ext

    def test_json_default(self):
        assert debug_dump._json_default(np.float32(0.5)) == 0.5
        assert debug_dump._json_default(np.array([1, 2])) == [1, 2]
        assert debug_dump._json_default(object).startswith("<class")

    def test_to_uint8_keeps_0_255_floats(self):
        out = debug_dump._to_uint8(np.array([[0.0, 128.0, 300.0]], np.float32))
        assert out.dtype == np.uint8 and out.tolist() == [[0, 128, 255]]
