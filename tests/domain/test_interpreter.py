"""Tests for the music theory interpreter."""

import json
from unittest.mock import MagicMock, patch

import pytest

from harmonia.domain.interpreter.prompts import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT
from harmonia.domain.interpreter.schemas import MusicParameters


class TestMusicParameters:
    def test_valid_params(self) -> None:
        p = MusicParameters(
            key="A", scale="minor", tempo_bpm=72.0, instruments=["Piano", "Bass"]
        )
        assert p.key == "A"
        assert p.scale == "minor"

    def test_scale_validation(self) -> None:
        with pytest.raises(Exception):
            MusicParameters(
                key="C", scale="dorian", tempo_bpm=100.0, instruments=["Piano"]
            )

    def test_tempo_bounds(self) -> None:
        with pytest.raises(Exception):
            MusicParameters(
                key="C", scale="major", tempo_bpm=300.0, instruments=["Piano"]
            )

    def test_instruments_non_empty(self) -> None:
        with pytest.raises(Exception):
            MusicParameters(key="C", scale="major", tempo_bpm=100.0, instruments=[])

    def test_conditioning_prompt_format(self) -> None:
        p = MusicParameters(
            key="D", scale="minor", tempo_bpm=65.0, instruments=["Piano", "Bass"]
        )
        prompt = p.conditioning_prompt
        assert "D minor" in prompt
        assert "65 BPM" in prompt
        assert "Piano" in prompt


class TestSystemPrompt:
    def test_prompt_contains_instrument_list(self) -> None:
        for inst in ["Piano", "Guitar", "Synth", "Bells", "Bass", "Flute", "Sax"]:
            assert inst in SYSTEM_PROMPT

    def test_few_shot_examples_are_valid_json(self) -> None:
        for ex in FEW_SHOT_EXAMPLES:
            data = json.loads(ex["assistant"])
            MusicParameters(**data)


class TestInterpretClient:
    def _mock_response(self, content: str) -> MagicMock:
        choice = MagicMock()
        choice.message.content = content
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    @patch("harmonia.domain.interpreter.client.OpenAI")
    def test_successful_interpret(self, mock_openai_cls: MagicMock) -> None:
        from harmonia.domain.interpreter.client import interpret

        payload = '{"key": "C", "scale": "major", "tempo_bpm": 120.0, "instruments": ["Piano"]}'
        mock_openai_cls.return_value.chat.completions.create.return_value = (
            self._mock_response(payload)
        )
        result = interpret("happy bright piece", api_key="x", base_url="http://x")
        assert result.key == "C"
        assert result.scale == "major"

    @patch("harmonia.domain.interpreter.client.OpenAI")
    def test_retries_on_bad_json(self, mock_openai_cls: MagicMock) -> None:
        from harmonia.domain.interpreter.client import interpret

        good = '{"key": "G", "scale": "major", "tempo_bpm": 100.0, "instruments": ["Piano"]}'
        mock_openai_cls.return_value.chat.completions.create.side_effect = [
            self._mock_response("not json"),
            self._mock_response(good),
        ]
        with patch("harmonia.domain.interpreter.client.time.sleep"):
            result = interpret("test", api_key="x", base_url="http://x")
        assert result.key == "G"

    @patch("harmonia.domain.interpreter.client.OpenAI")
    def test_raises_after_max_retries(self, mock_openai_cls: MagicMock) -> None:
        from harmonia.domain.interpreter.client import interpret

        mock_openai_cls.return_value.chat.completions.create.return_value = (
            self._mock_response("definitely not json")
        )
        with patch("harmonia.domain.interpreter.client.time.sleep"):
            with pytest.raises(ValueError, match="invalid MusicParameters"):
                interpret("test", api_key="x", base_url="http://x")
