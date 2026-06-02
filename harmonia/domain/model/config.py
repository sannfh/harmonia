"""LoRA fine-tuning config and inference hyperparameters."""

from dataclasses import dataclass, field


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    fp16: bool = True
    logging_steps: int = 100
    save_steps: int = 1000
    save_total_limit: int = 3
    optim: str = "paged_adamw_8bit"
    max_seq_length: int = 1024


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 0.9
    top_p: float = 0.95
    repetition_penalty: float = 1.1


LORA = LoRAConfig()
QUANTIZATION = QuantizationConfig()
TRAINING = TrainingConfig()
GENERATION = GenerationConfig()

BASE_MODEL = "meta-llama/Llama-3.2-3B"
MIDI_START_TOKEN = "<MIDI_START>"
MIDI_END_TOKEN = "<MIDI_END>"
