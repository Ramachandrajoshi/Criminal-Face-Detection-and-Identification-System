"""
Application start-up (app/main.py): GPU configuration fallbacks, model
warm-up, start-up migrations and the OpenAPI security scheme.
"""

import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app import main


class TestConfigureGpuFallbacks:
    def test_no_gpu_detected_leaves_tensorflow_untouched(self):
        tf = MagicMock()
        tf.config.list_physical_devices.return_value = []
        with patch.dict(sys.modules, {"tensorflow": tf}), patch("app.main.settings.enable_gpu", True):
            main._configure_gpu()
        tf.config.set_visible_devices.assert_not_called()
        tf.config.experimental.set_memory_growth.assert_not_called()

    def test_tensorflow_failure_never_crashes_startup(self):
        tf = MagicMock()
        tf.config.list_physical_devices.side_effect = RuntimeError("CUDA_ERROR_INVALID_HANDLE")
        with patch.dict(sys.modules, {"tensorflow": tf}), patch("app.main.settings.enable_gpu", True):
            main._configure_gpu()   # logs and returns


def _deepface(build_model=None):
    return SimpleNamespace(DeepFace=SimpleNamespace(build_model=build_model or MagicMock()))


class TestWarmup:
    def test_deepface_recipes_warm_model_and_detectors(self):
        deepface = _deepface()
        with patch.dict(sys.modules, {"deepface": deepface}), \
             patch("app.core.pipeline.detect_face") as detect, \
             patch("app.main.CURRENT_EMBEDDING_VERSION", 3), \
             patch("app.core.face_engine.get_engine") as get_engine:
            main._warmup_models()

        deepface.DeepFace.build_model.assert_called_once_with(main.settings.deepface_model)
        detect.assert_called_once()
        assert detect.call_args.args[0][:3] == b"\xff\xd8\xff"     # a JPEG-encoded dummy frame
        get_engine.assert_not_called()                              # version 4 not active

    def test_version_4_runs_every_detector_size_and_the_embedder(self):
        engine = MagicMock(det_size=640, providers=["CPUExecutionProvider"])
        with patch.dict(sys.modules, {"deepface": _deepface()}), \
             patch("app.core.pipeline.detect_face"), \
             patch("app.main.CURRENT_EMBEDDING_VERSION", main.INSIGHTFACE_EMBEDDING_VERSION), \
             patch("app.core.face_engine.get_engine", return_value=engine):
            main._warmup_models()

        sizes = [c.kwargs.get("size") for c in engine.detect.call_args_list]
        assert sizes == [None, 1280]       # standard pass, then the "large" fallback size
        assert engine.detect.call_args_list[0].args[0].shape == (640, 640, 3)
        assert engine.embed.call_args.args[0].shape == (112, 112, 3)

    def test_failures_are_logged_never_raised(self):
        failing = _deepface(build_model=MagicMock(side_effect=OSError("weights missing")))
        with patch.dict(sys.modules, {"deepface": failing}), \
             patch("app.core.pipeline.detect_face", side_effect=RuntimeError("detector broke")), \
             patch("app.main.CURRENT_EMBEDDING_VERSION", main.INSIGHTFACE_EMBEDDING_VERSION), \
             patch("app.core.face_engine.get_engine", side_effect=FileNotFoundError("no onnx")):
            main._warmup_models()


@asynccontextmanager
async def _fake_session():
    yield MagicMock()


class TestLifespan:
    @pytest.mark.parametrize("summary_or_error", [
        {"applied": 1, "skipped": 8, "errors": []},
        {"applied": 0, "skipped": 0, "errors": [{"migration": "x", "error": "bad DDL"}]},
        RuntimeError("database unreachable"),
    ])
    async def test_runs_gpu_setup_warmup_and_migrations(self, summary_or_error):
        if isinstance(summary_or_error, Exception):
            migrations = AsyncMock(side_effect=summary_or_error)
        else:
            migrations = AsyncMock(return_value=summary_or_error)

        with patch("app.main._configure_gpu") as gpu, \
             patch("app.main._warmup_models") as warmup, \
             patch("app.main.async_session_factory", _fake_session), \
             patch("app.main.run_migrations", migrations):
            async with main.lifespan(main.app):
                pass                          # the app serves requests here

        gpu.assert_called_once()
        warmup.assert_called_once()
        migrations.assert_awaited_once()


class TestOpenApi:
    def test_schema_declares_bearer_auth_globally_and_is_cached(self):
        main.app.openapi_schema = None
        try:
            schema = main.app.openapi()
            assert schema["components"]["securitySchemes"]["BearerAuth"]["scheme"] == "bearer"
            assert schema["security"] == [{"BearerAuth": []}]
            assert "/api/v1/faces/{face_id}" in schema["paths"]
            assert main.app.openapi() is schema
        finally:
            main.app.openapi_schema = None

