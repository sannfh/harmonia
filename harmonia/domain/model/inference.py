"""HarmoniaGenerator: load fine-tuned LLaMA, generate REMI tokens, decode to MIDI."""

from pathlib import Path
from typing import Any

import pretty_midi
from miditok import REMI, TokenizerConfig

from harmonia.domain.model.config import GENERATION, MIDI_END_TOKEN, MIDI_START_TOKEN


def _build_tokenizer() -> REMI:
    config = TokenizerConfig(
        num_velocities=32,
        use_chords=False,
        use_programs=True,
        use_time_signatures=False,
    )
    return REMI(config)


class HarmoniaGenerator:
    """Wraps a fine-tuned LLaMA model for MIDI generation."""

    def __init__(self, model_repo: str) -> None:
        self._model_repo = model_repo
        self._model = None
        self._tokenizer = None
        self._midi_tokenizer = _build_tokenizer()

    def _load(self) -> None:
        """Lazy-load model weights on first use."""
        # Heavy imports kept here so the module is importable without torch installed
        import torch  # type: ignore[import-untyped]
        from transformers import (  # type: ignore[import-untyped]
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_repo)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_repo,
            quantization_config=bnb,
            device_map="auto",
        )
        self._model.eval()

    def generate_midi(
        self,
        conditioning_prompt: str,
        output_path: Path,
    ) -> Path:
        """Generate a MIDI file from a conditioning prompt.

        Args:
            conditioning_prompt: Natural-language conditioning string produced by
                ``build_conditioning_prompt``.
            output_path: Where to write the resulting ``.mid`` file.

        Returns:
            The path to the written MIDI file.
        """
        if self._model is None:
            self._load()

        import torch  # type: ignore[import-untyped]

        prompt = f"{conditioning_prompt}\n{MIDI_START_TOKEN}\n"
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)  # type: ignore[union-attr]

        with torch.no_grad():
            output_ids = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=GENERATION.max_new_tokens,
                do_sample=GENERATION.do_sample,
                temperature=GENERATION.temperature,
                top_p=GENERATION.top_p,
                repetition_penalty=GENERATION.repetition_penalty,
                pad_token_id=self._tokenizer.eos_token_id,  # type: ignore[union-attr]
            )

        generated = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)  # type: ignore[union-attr]
        remi_tokens = _extract_remi_tokens(generated)
        if not remi_tokens:
            raise ValueError("Model produced no MIDI tokens between markers.")

        midi = _decode_remi_to_midi(remi_tokens, self._midi_tokenizer)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        midi.write(str(output_path))
        return output_path


def _extract_remi_tokens(generated_text: str) -> list[str]:
    """Pull the token list between <MIDI_START> and <MIDI_END>."""
    start = generated_text.find(MIDI_START_TOKEN)
    if start == -1:
        return []
    after_start = generated_text[start + len(MIDI_START_TOKEN) :].lstrip("\n")

    end = after_start.find(MIDI_END_TOKEN)
    token_str = after_start[:end].strip() if end != -1 else after_start.strip()
    return token_str.split() if token_str else []


def _decode_remi_to_midi(
    tokens: list[str],
    tokenizer: REMI,
) -> pretty_midi.PrettyMIDI:
    """Convert a list of REMI token strings back to a PrettyMIDI object."""
    vocab: Any = tokenizer.vocab
    ids = [vocab[t] for t in tokens if t in vocab]
    if not ids:
        raise ValueError("No valid REMI tokens found after vocabulary lookup.")

    from miditok import TokSequence  # type: ignore[import-untyped]

    seq = TokSequence(ids=ids)
    tokenizer.complete_sequence(seq)
    midi = tokenizer.tokens_to_midi([seq])
    return midi
