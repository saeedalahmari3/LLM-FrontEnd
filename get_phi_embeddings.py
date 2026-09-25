import atexit
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ollama import ResponseError, embed, list as ollama_list, pull


DEFAULT_EMBEDDING_MODEL = "phi4-mini"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PHI4_EMBEDDING_HOST = "127.0.0.1"
DEFAULT_PHI4_EMBEDDING_PORT = 11435
DEFAULT_PHI4_EMBEDDING_BATCH_SIZE = 1024
DEFAULT_PHI4_EMBEDDING_CONTEXT_LENGTH = 4096
DEFAULT_PHI4_EMBEDDING_MAX_BATCH_SIZE = 8192
DEFAULT_LLAMA_SERVER_PATH = (
    "/Applications/Ollama.app/Contents/Resources/llama-server"
)
EMPTY_TEXT_PLACEHOLDER = "[EMPTY RESPONSE]"

_sidecar_lock = threading.Lock()
_sidecar_process = None
_sidecar_port = None
_sidecar_batch_size = None
_sidecar_device = None
_use_phi4_sidecar = False


@lru_cache(maxsize=None)
def _get_sentence_transformer(model_name):
    """Load the local sentence-embedding model only when it is first needed."""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(
            model_name,
            local_files_only=True,
        )
    except OSError as exc:
        raise RuntimeError(
            f"The cached embedding model '{model_name}' is unavailable. "
            "Download it once while online, or set EMBEDDING_BACKEND=ollama."
        ) from exc


def _model_name(model):
    """Read a model name from either an Ollama model object or a dictionary."""
    if isinstance(model, dict):
        return model.get("model") or model.get("name")
    return getattr(model, "model", None) or getattr(model, "name", None)


def _stop_phi4_embedding_sidecar():
    """Stop the embedding server only when this Python process started it."""
    global _sidecar_port, _sidecar_process

    if _sidecar_process is not None and _sidecar_process.poll() is None:
        _sidecar_process.terminate()
        try:
            _sidecar_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _sidecar_process.kill()
    _sidecar_process = None
    if not os.getenv("PHI4_EMBEDDING_PORT"):
        _sidecar_port = None


atexit.register(_stop_phi4_embedding_sidecar)


def _parse_positive_int(value, name):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, not {value!r}.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, not {value!r}.")
    return parsed


def _next_power_of_two(value):
    return 1 << (int(value) - 1).bit_length()


def _find_free_local_port():
    host = os.getenv("PHI4_EMBEDDING_HOST", DEFAULT_PHI4_EMBEDDING_HOST)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _get_phi4_sidecar_port():
    global _sidecar_port

    configured_port = os.getenv("PHI4_EMBEDDING_PORT")
    if configured_port:
        _sidecar_port = _parse_positive_int(
            configured_port,
            "PHI4_EMBEDDING_PORT",
        )
    elif _sidecar_port is None:
        _sidecar_port = _find_free_local_port()
    return _sidecar_port


def _get_phi4_sidecar_batch_size():
    configured_batch_size = _parse_positive_int(
        os.getenv(
            "PHI4_EMBEDDING_BATCH_SIZE",
            str(DEFAULT_PHI4_EMBEDDING_BATCH_SIZE),
        ),
        "PHI4_EMBEDDING_BATCH_SIZE",
    )
    return _sidecar_batch_size or configured_batch_size


def _get_phi4_sidecar_device():
    return _sidecar_device or os.getenv("PHI4_EMBEDDING_DEVICE", "auto")


def _phi4_sidecar_url(path):
    host = os.getenv("PHI4_EMBEDDING_HOST", DEFAULT_PHI4_EMBEDDING_HOST)
    port = _get_phi4_sidecar_port()
    return f"http://{host}:{port}{path}"


def _normalize_embedding_text(text):
    """Convert rejected empty model outputs into an embeddable placeholder."""
    normalized = str(text).replace("\x00", " ").strip()
    return normalized or EMPTY_TEXT_PLACEHOLDER


def _sidecar_is_ready():
    try:
        with urlopen(_phi4_sidecar_url("/health"), timeout=1) as response:
            return response.status == 200
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


def _find_llama_server():
    configured_path = os.getenv("OLLAMA_LLAMA_SERVER")
    candidates = [
        configured_path,
        DEFAULT_LLAMA_SERVER_PATH,
        shutil.which("llama-server"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate))

    raise RuntimeError(
        "Could not find Ollama's llama-server executable. Set "
        "OLLAMA_LLAMA_SERVER to its full path."
    )


def _find_ollama_model_file(model_name):
    """Resolve the GGUF path for an installed Ollama model."""
    try:
        result = subprocess.run(
            ["ollama", "show", "--modelfile", model_name],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Could not resolve the installed Ollama model '{model_name}'."
        ) from exc

    for line in result.stdout.splitlines():
        if line.startswith("FROM "):
            model_path = shlex.split(line[5:].strip())[0]
            if Path(model_path).is_file():
                return model_path

    raise RuntimeError(
        f"Ollama did not return a readable GGUF path for '{model_name}'."
    )


def _start_phi4_embedding_sidecar(model_name):
    """Start phi4-mini with Ollama's bundled llama-server embedding mode."""
    global _sidecar_process

    _get_phi4_sidecar_port()
    if _sidecar_is_ready():
        return

    with _sidecar_lock:
        if _sidecar_is_ready():
            return
        if _sidecar_process is not None and _sidecar_process.poll() is None:
            return

        llama_server = _find_llama_server()
        model_file = _find_ollama_model_file(model_name)
        host = os.getenv(
            "PHI4_EMBEDDING_HOST",
            DEFAULT_PHI4_EMBEDDING_HOST,
        )
        port = str(_get_phi4_sidecar_port())
        batch_size = str(_get_phi4_sidecar_batch_size())
        context_length = str(
            max(
                _parse_positive_int(
                    os.getenv(
                        "PHI4_EMBEDDING_CONTEXT_LENGTH",
                        str(DEFAULT_PHI4_EMBEDDING_CONTEXT_LENGTH),
                    ),
                    "PHI4_EMBEDDING_CONTEXT_LENGTH",
                ),
                _get_phi4_sidecar_batch_size(),
            )
        )

        command = [
            llama_server,
            "--model",
            model_file,
            "--embeddings",
            "--pooling",
            "mean",
            "--host",
            host,
            "--port",
            port,
            "--ctx-size",
            context_length,
            "--batch-size",
            batch_size,
            "--ubatch-size",
            batch_size,
            "--alias",
            model_name,
            "--flash-attn",
            "off",
            "--no-ui",
        ]
        device = _get_phi4_sidecar_device()
        if device != "auto":
            command.extend(["--device", device])

        _sidecar_process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        timeout = float(os.getenv("PHI4_EMBEDDING_STARTUP_TIMEOUT", "120"))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _sidecar_is_ready():
                return
            if _sidecar_process.poll() is not None:
                raise RuntimeError(
                    "The phi4-mini embedding sidecar exited during startup."
                )
            time.sleep(0.25)

        _stop_phi4_embedding_sidecar()
        raise RuntimeError(
            f"Timed out after {timeout:g} seconds while starting the "
            "phi4-mini embedding sidecar."
        )


def _too_large_token_count(error_detail):
    match = re.search(r"input \((\d+) tokens\) is too large", error_detail)
    if match:
        return int(match.group(1))
    return None


def _is_compute_error(error_detail):
    return "Compute error" in error_detail


def _restart_phi4_embedding_sidecar_with_larger_batch(model_name, token_count):
    global _sidecar_batch_size

    current_batch_size = _get_phi4_sidecar_batch_size()
    target_batch_size = _next_power_of_two(
        max(token_count + 128, current_batch_size * 2)
    )
    max_batch_size = _parse_positive_int(
        os.getenv(
            "PHI4_EMBEDDING_MAX_BATCH_SIZE",
            str(DEFAULT_PHI4_EMBEDDING_MAX_BATCH_SIZE),
        ),
        "PHI4_EMBEDDING_MAX_BATCH_SIZE",
    )
    if target_batch_size > max_batch_size:
        raise RuntimeError(
            f"phi4-mini embedding input needs about {token_count} tokens, "
            f"which exceeds PHI4_EMBEDDING_MAX_BATCH_SIZE={max_batch_size}. "
            "Increase PHI4_EMBEDDING_MAX_BATCH_SIZE or shorten the input text."
        )

    _sidecar_batch_size = target_batch_size
    _stop_phi4_embedding_sidecar()
    _start_phi4_embedding_sidecar(model_name)


def _restart_phi4_embedding_sidecar_on_cpu(model_name):
    global _sidecar_batch_size, _sidecar_device

    _sidecar_device = "none"
    _sidecar_batch_size = min(
        _get_phi4_sidecar_batch_size(),
        DEFAULT_PHI4_EMBEDDING_BATCH_SIZE,
    )
    _stop_phi4_embedding_sidecar()
    _start_phi4_embedding_sidecar(model_name)


def _request_phi4_sidecar_embeddings(texts, model_name):
    payload = json.dumps(
        {
            "model": model_name,
            "input": texts,
        }
    ).encode("utf-8")
    request = Request(
        _phi4_sidecar_url("/v1/embeddings"),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        error_detail = error_body or str(exc)
        raise RuntimeError(
            "The phi4-mini embedding sidecar rejected the request "
            f"(HTTP {exc.code}): {error_detail}"
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"The phi4-mini embedding sidecar request failed: {exc}"
        ) from exc

    data = sorted(result["data"], key=lambda item: item["index"])
    return [item["embedding"] for item in data]


def _get_phi4_sidecar_embeddings(
    texts,
    model_name,
    allow_size_retry=True,
    allow_split_retry=True,
    allow_cpu_retry=True,
):
    _start_phi4_embedding_sidecar(model_name)
    try:
        return _request_phi4_sidecar_embeddings(texts, model_name)
    except RuntimeError as exc:
        error_detail = str(exc)
        token_count = _too_large_token_count(error_detail)
        if token_count is not None and allow_size_retry:
            _restart_phi4_embedding_sidecar_with_larger_batch(model_name, token_count)
            return _get_phi4_sidecar_embeddings(
                texts,
                model_name,
                allow_size_retry=False,
                allow_split_retry=allow_split_retry,
                allow_cpu_retry=allow_cpu_retry,
            )

        if _is_compute_error(error_detail):
            if allow_split_retry and len(texts) > 1:
                embeddings = []
                for text in texts:
                    embeddings.extend(
                        _get_phi4_sidecar_embeddings(
                            [text],
                            model_name,
                            allow_size_retry=allow_size_retry,
                            allow_split_retry=False,
                            allow_cpu_retry=allow_cpu_retry,
                        )
                    )
                return embeddings

            if allow_cpu_retry and _get_phi4_sidecar_device() != "none":
                _restart_phi4_embedding_sidecar_on_cpu(model_name)
                return _get_phi4_sidecar_embeddings(
                    texts,
                    model_name,
                    allow_size_retry=allow_size_retry,
                    allow_split_retry=False,
                    allow_cpu_retry=False,
                )

        raise


@lru_cache(maxsize=None)
def _ensure_model_is_available(model_name):
    """Pull the embedding model once when it is not installed locally."""
    models_response = ollama_list()
    models = (
        models_response.get("models", [])
        if isinstance(models_response, dict)
        else getattr(models_response, "models", [])
    )
    available_models = {_model_name(model) for model in models}
    available_models.discard(None)

    requested_base_name = model_name.removesuffix(":latest")
    available_base_names = {
        available_model.removesuffix(":latest")
        for available_model in available_models
    }

    if requested_base_name not in available_base_names:
        print(f"Embedding model {model_name} not found. Pulling it with Ollama...")
        pull(model_name)


def get_phi_embeddings(inputs, model_name=None):
    """Return embeddings for one string or a sequence of strings.

    By default this uses the Ollama ``phi4-mini`` model. If Ollama's main API
    does not expose embeddings for it, this module starts Ollama's bundled
    llama-server in embedding mode against the same installed GGUF. Set
    ``EMBEDDING_BACKEND=sentence-transformers`` to use the cached local
    Sentence Transformer model instead. ``OLLAMA_EMBED_MODEL`` can override
    the default Ollama model.
    """
    texts = [inputs] if isinstance(inputs, str) else list(inputs)
    if not texts:
        return []

    backend = os.getenv("EMBEDDING_BACKEND", "ollama").lower()
    if backend in {"sentence-transformers", "sentence_transformers", "local"}:
        sentence_transformer_model = (
            model_name
            or os.getenv("SENTENCE_TRANSFORMER_MODEL")
            or DEFAULT_SENTENCE_TRANSFORMER_MODEL
        )
        model = _get_sentence_transformer(sentence_transformer_model)
        return model.encode(
            [str(text) for text in texts],
            show_progress_bar=False,
        ).tolist()

    if backend != "ollama":
        raise ValueError(
            "EMBEDDING_BACKEND must be 'sentence-transformers' or 'ollama', "
            f"not {backend!r}."
        )

    model_name = (
        model_name
        or os.getenv("OLLAMA_EMBED_MODEL")
        or DEFAULT_EMBEDDING_MODEL
    )
    _ensure_model_is_available(model_name)

    texts = [_normalize_embedding_text(text) for text in texts]
    global _use_phi4_sidecar
    if _use_phi4_sidecar and model_name.removesuffix(":latest") == "phi4-mini":
        return _get_phi4_sidecar_embeddings(texts, model_name)

    try:
        response = embed(model=model_name, input=texts)
    except ResponseError as exc:
        if (
            exc.status_code == 501
            and model_name.removesuffix(":latest") == "phi4-mini"
        ):
            _use_phi4_sidecar = True
            return _get_phi4_sidecar_embeddings(texts, model_name)
        raise

    return response["embeddings"]


#print(get_phi_embeddings('I am Saeed Alahmari'))
