"""Tests for the MIDI generation model (inference module)."""

from harmonia.domain.model.inference import _extract_remi_tokens


class TestExtractRemiTokens:
    def test_extracts_between_markers(self) -> None:
        text = "Compose a piece.\n<MIDI_START>\nBar_1 Tempo_120 Note_On\n<MIDI_END>"
        tokens = _extract_remi_tokens(text)
        assert tokens == ["Bar_1", "Tempo_120", "Note_On"]

    def test_no_start_marker_returns_empty(self) -> None:
        tokens = _extract_remi_tokens("some text without markers")
        assert tokens == []

    def test_no_end_marker_returns_tokens_to_end(self) -> None:
        text = "<MIDI_START>\nBar_1 Note_On"
        tokens = _extract_remi_tokens(text)
        assert tokens == ["Bar_1", "Note_On"]

    def test_empty_token_section_returns_empty(self) -> None:
        text = "<MIDI_START>\n\n<MIDI_END>"
        tokens = _extract_remi_tokens(text)
        assert tokens == []


class TestHarmoniaGeneratorLazyLoad:
    def test_model_not_loaded_at_init(self) -> None:
        from harmonia.domain.model.inference import HarmoniaGenerator

        gen = HarmoniaGenerator("some/repo")
        assert gen._model is None
        assert gen._tokenizer is None
