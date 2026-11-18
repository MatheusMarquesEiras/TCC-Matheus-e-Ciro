"""Classificador de especialidade sobre um modelo treinado.

Encapsula o ciclo montar mensagens → aplicar chat template → gerar → casar o
texto gerado com um rótulo válido, para que avaliação e demo compartilhem
exatamente o mesmo caminho de inferência.
"""

# modeling importa unsloth, que precisa vir antes de transformers.
from .modeling import apply_chat_template, load_model, text_tokenizer

import torch

from .model_registry import ModelSpec
from .prompting import build_inference_messages, match_label

#: Rótulos são curtos; 20 tokens bastam e evitam divagação do modelo.
MAX_NEW_TOKENS = 20


class Classifier:
    """Modelo carregado e pronto para classificar queixas."""

    def __init__(self, model, tokenizer, spec: ModelSpec, labels: list[str], system_template=None):
        self.model = model
        self.tokenizer = tokenizer
        self.spec = spec
        self.labels = labels
        self.system_template = system_template

    @classmethod
    def load(
        cls,
        spec: ModelSpec,
        labels: list[str],
        model_dir: str | None = None,
        system_template: str | None = None,
    ) -> "Classifier":
        """Carrega o modelo base (``model_dir=None``) ou os adaptadores treinados."""
        model, tokenizer = load_model(spec, model_dir=model_dir, for_inference=True)
        return cls(model, tokenizer, spec, labels, system_template)

    def _template_kwargs(self) -> dict:
        return {"template": self.system_template} if self.system_template else {}

    @torch.inference_mode()
    def predict(
        self,
        question: str,
        few_shot: list[dict] | None = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> str:
        """Classifica uma queixa e devolve um rótulo válido do dataset."""
        messages = build_inference_messages(
            question, self.labels, few_shot, **self._template_kwargs()
        )
        text = apply_chat_template(
            self.tokenizer, messages, self.spec, add_generation_prompt=True
        )
        tok = text_tokenizer(self.tokenizer)
        inputs = tok(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        generated = tok.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return match_label(generated, self.labels)

    def release(self) -> None:
        """Libera a VRAM — necessário entre setups na avaliação."""
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
