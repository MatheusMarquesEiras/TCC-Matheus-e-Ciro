"""Backends de LLM-grader para a avaliação qualitativa.

Três opções, com a mesma interface ``grade(system, prompt) -> str``:

``ollama``        modelo local (``ollama serve`` em localhost)
``ollama-cloud``  modelos grandes hospedados pela Ollama, mesma API HTTP com
                  ``Authorization: Bearer``. Serve para julgar os modelos que
                  ajustamos sem depender do que cabe na GPU de 12 GB.
``anthropic``     Claude via SDK oficial

Cloud e local falam o mesmo endpoint ``/api/chat``: o que muda é o host e o
cabeçalho de autenticação, então a mesma classe atende os dois.
"""

import json
import os
import time
import urllib.error
import urllib.request

OLLAMA_LOCAL_HOST = "http://localhost:11434"
OLLAMA_CLOUD_HOST = "https://ollama.com"

DEFAULT_LOCAL_MODEL = "llama3.1:8b"
DEFAULT_CLOUD_MODEL = "gpt-oss:120b"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"

#: Falhas de rede acontecem em avaliações longas; repetir é mais barato que
#: perder a rodada inteira.
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 3


class GraderError(RuntimeError):
    """Falha irrecuperável ao consultar o grader."""


class OllamaGrader:
    """Grader via API HTTP do Ollama — servidor local ou nuvem."""

    def __init__(
        self,
        model: str,
        host: str = OLLAMA_LOCAL_HOST,
        api_key: str | None = None,
        timeout: int = 180,
        temperature: float = 0.0,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature
        self.is_cloud = self.host.rstrip("/") != OLLAMA_LOCAL_HOST and "localhost" not in self.host

        if self.is_cloud and not self.api_key:
            raise GraderError(
                "Ollama Cloud exige uma chave de API. Informe --api_key ou defina a "
                "variável de ambiente OLLAMA_API_KEY. "
                "A chave é criada em https://ollama.com/settings/keys"
            )
        if not self.is_cloud:
            self._check_local_server()

    @property
    def label(self) -> str:
        return f"Ollama {'Cloud' if self.is_cloud else 'local'} ({self.model})"

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _check_local_server(self) -> None:
        try:
            request = urllib.request.Request(f"{self.host}/api/tags", headers=self._headers())
            with urllib.request.urlopen(request, timeout=10):
                pass
        except (urllib.error.URLError, OSError) as error:
            raise GraderError(
                f"Não foi possível conectar ao Ollama em {self.host}. "
                f"Verifique se o servidor está rodando (`ollama serve`). Erro: {error}"
            ) from error

    def grade(self, system: str, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": self.temperature},
            }
        ).encode("utf-8")

        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            request = urllib.request.Request(
                f"{self.host}/api/chat",
                data=payload,
                headers=self._headers(),
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return data["message"]["content"]
            except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as error:
                last_error = error
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)

        raise GraderError(
            f"Falha ao consultar {self.label} após {MAX_RETRIES} tentativas: {last_error}"
        ) from last_error


class AnthropicGrader:
    """Grader via API da Anthropic."""

    def __init__(self, api_key: str, model: str = DEFAULT_ANTHROPIC_MODEL, max_tokens: int = 1024):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    @property
    def label(self) -> str:
        return f"Anthropic ({self.model})"

    def grade(self, system: str, prompt: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as error:  # noqa: BLE001 — erros do SDK variam
                last_error = error
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        raise GraderError(
            f"Falha ao consultar {self.label} após {MAX_RETRIES} tentativas: {last_error}"
        ) from last_error


def build_grader(
    backend: str,
    model: str | None = None,
    host: str | None = None,
    api_key: str | None = None,
    timeout: int = 180,
):
    """Instancia o grader escolhido, preenchendo host, modelo e chave padrão."""
    if backend == "ollama":
        return OllamaGrader(
            model=model or DEFAULT_LOCAL_MODEL,
            host=host or OLLAMA_LOCAL_HOST,
            api_key=api_key or os.getenv("OLLAMA_API_KEY"),
            timeout=timeout,
        )
    if backend == "ollama-cloud":
        return OllamaGrader(
            model=model or DEFAULT_CLOUD_MODEL,
            host=host or OLLAMA_CLOUD_HOST,
            api_key=api_key or os.getenv("OLLAMA_API_KEY"),
            timeout=timeout,
        )
    if backend == "anthropic":
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise GraderError(
                "Informe a chave em --api_key ou na variável ANTHROPIC_API_KEY."
            )
        return AnthropicGrader(api_key=key, model=model or DEFAULT_ANTHROPIC_MODEL)
    raise GraderError(
        f"Backend de grader '{backend}' desconhecido. "
        "Use 'ollama', 'ollama-cloud' ou 'anthropic'."
    )
