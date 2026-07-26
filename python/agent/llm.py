import backoff
import os
from typing import Tuple
import requests
import litellm
from dotenv import load_dotenv
import json

load_dotenv()

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))

# --- llama.cpp server (direct GGUF, OpenAI-compatible API) ---
# Start: llama-server -m model.gguf --port 8080 -c 8192 -ngl 99
# Use:   MODEL_NAME=llamacpp/gemma-4-31B-it-Q4_K_M  (label after / is cosmetic)
LLAMACPP_BASE_URL = os.environ.get("LLAMACPP_BASE_URL", "http://localhost:8080")
LLAMACPP_GEMMA4   = "llamacpp/gemma-4-31B-it-Q4_K_M"

# --- Ollama (local) models ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLAMA = "ollama/llama3.2"
OLLAMA_CODELLAMA = "ollama/codellama"
OLLAMA_MISTRAL = "ollama/mistral"
OLLAMA_DEEPSEEK = "ollama/deepseek-coder-v2"
OLLAMA_QWEN = "ollama/qwen2.5-coder"

# --- MLX (Apple Silicon local) models ---
# Use "mlx/<model-name-or-path>" to run models locally via mlx-lm on Mac.
# The part after "mlx/" is passed directly to mlx_lm.load() — it can be a
# HuggingFace repo ID or a local path to a downloaded MLX model.
MLX_MODEL_PATH = os.environ.get("MLX_MODEL_PATH", "")  # optional local path override
MLX_QWEN_OPUS = "mlx/BeastCode/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"

# --- OpenRouter (cloud gateway — 300+ models, many free) ---
# Use "openrouter/<provider>/<model>" e.g. openrouter/google/gemma-3-4b-it:free
# Set OPENROUTER_API_KEY in .env. Free tier models end in :free.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OR_GEMMA4_FREE    = "openrouter/google/gemma-3-4b-it:free"
OR_GEMMA4         = "openrouter/google/gemma-3-4b-it"
OR_LLAMA4_FREE    = "openrouter/meta-llama/llama-4-scout:free"
OR_LLAMA4         = "openrouter/meta-llama/llama-4-scout"
OR_QWEN3_FREE     = "openrouter/qwen/qwen3-8b:free"
OR_DEEPSEEK_FREE  = "openrouter/deepseek/deepseek-r1-0528:free"
OR_CLAUDE_SONNET  = "openrouter/anthropic/claude-sonnet-4-5"
OR_GPT4O          = "openrouter/openai/gpt-4o"

# --- Cloud models (original) ---
CLAUDE_MODEL = "anthropic/claude-sonnet-4-5-20250929"
CLAUDE_HAIKU_MODEL = "anthropic/claude-3-haiku-20240307"
CLAUDE_35NEW_MODEL = "anthropic/claude-3-5-sonnet-20241022"
OPENAI_MODEL = "openai/gpt-4o"
OPENAI_MINI_MODEL = "openai/gpt-4o-mini"
OPENAI_O3_MODEL = "openai/o3"
OPENAI_O3MINI_MODEL = "openai/o3-mini"
OPENAI_O4MINI_MODEL = "openai/o4-mini"
OPENAI_GPT52_MODEL = "openai/gpt-5.2"
OPENAI_GPT5_MODEL = "openai/gpt-5"
OPENAI_GPT5MINI_MODEL = "openai/gpt-5-mini"
GEMINI_3_MODEL = "gemini/gemini-3-pro-preview"
GEMINI_MODEL = "gemini/gemini-2.5-pro"
GEMINI_FLASH_MODEL = "gemini/gemini-2.5-flash"

# --- Default model (configurable via env) ---
DEFAULT_MODEL = os.environ.get("MODEL_NAME", OLLAMA_LLAMA)

litellm.drop_params=True


# ---------------------------------------------------------------------------
# MLX helper — lazily loaded so mlx-lm is only required when using mlx/ models
# ---------------------------------------------------------------------------
_mlx_model_cache = {}  # cache loaded model+tokenizer by path


def _get_mlx_response(messages, model_id, max_tokens, temperature):
    """Generate a response using mlx-lm (Apple Silicon only)."""
    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        raise ImportError(
            "mlx-lm is required for MLX models. Install it with: "
            "pip install mlx-lm"
        )

    # Resolve model path: env override, local path, or HuggingFace repo
    model_name = model_id.removeprefix("mlx/")
    model_path = MLX_MODEL_PATH or model_name

    # Cache model + tokenizer across calls for performance
    if model_path not in _mlx_model_cache:
        print(f"  [MLX] Loading model from {model_path} …")
        _mlx_model_cache[model_path] = load(model_path)

    model_obj, tokenizer = _mlx_model_cache[model_path]

    # Build chat prompt via the tokenizer's chat template
    chat_kwargs = dict(
        tokenize=False,
        add_generation_prompt=True,
    )
    # Enable thinking/reasoning if the tokenizer supports it
    try:
        prompt = tokenizer.apply_chat_template(
            messages, **chat_kwargs, enable_thinking=True
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)

    sampler = make_sampler(temp=temperature)
    response_text = generate(
        model_obj,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=True,
    )
    return response_text


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, json.JSONDecodeError, KeyError),
    max_time=600,
    max_value=60,
)
def get_response_from_llm(
    msg: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    msg_history=None,
) -> Tuple[str, list, dict]:
    if msg_history is None:
        msg_history = []

    # Convert text to content, compatible with LITELLM API
    msg_history = [
        {**msg, "content": msg["text"]} if "text" in msg else dict(msg)
        for msg in msg_history
    ]

    new_msg_history = msg_history + [{"role": "user", "content": msg}]

    # ----- llama.cpp server (OpenAI-compatible, no key required) -----
    if model.startswith("llamacpp/"):
        # The model label after "llamacpp/" is cosmetic — the server uses
        # whatever GGUF was loaded at startup, so we just pass it as-is.
        model_label = model.removeprefix("llamacpp/") or "local"
        api_url = LLAMACPP_BASE_URL.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": model_label,
            "messages": new_msg_history,
            "max_tokens": min(max_tokens, 4096),
            "temperature": temperature,
        }
        resp = requests.post(api_url, json=payload, timeout=600)
        resp.raise_for_status()
        response_text = resp.json()["choices"][0]["message"]["content"]
        new_msg_history.append({"role": "assistant", "content": response_text})
        new_msg_history = [
            {**m, "text": m["content"]} if "content" in m else dict(m)
            for m in new_msg_history
        ]
        return response_text, new_msg_history, {}

    # ----- MLX path (Apple Silicon local inference) -----
    if model.startswith("mlx/"):
        response_text = _get_mlx_response(
            new_msg_history, model, max_tokens=min(max_tokens, 4096),
            temperature=temperature,
        )
        new_msg_history.append({"role": "assistant", "content": response_text})
        # Convert content→text for MetaGen API compatibility
        new_msg_history = [
            {**m, "text": m["content"]} if "content" in m else dict(m)
            for m in new_msg_history
        ]
        return response_text, new_msg_history, {}

    # ----- litellm path (Ollama / Cloud) -----
    # Build kwargs - handle model-specific requirements
    completion_kwargs = {
        "model": model,
        "messages": new_msg_history,
    }

    # Set api_base / api_key for Ollama and OpenRouter
    if model.startswith("ollama/") or model.startswith("ollama_chat/"):
        completion_kwargs["api_base"] = OLLAMA_BASE_URL
    elif model.startswith("openrouter/"):
        completion_kwargs["api_base"] = OPENROUTER_BASE_URL
        completion_kwargs["api_key"] = OPENROUTER_API_KEY

    # GPT-5 and GPT-5-mini only support default temperature (1), skip it
    # GPT-5.2 supports temperature
    if model in ["openai/gpt-5", "openai/gpt-5-mini"]:
        pass  # Don't set temperature
    else:
        completion_kwargs["temperature"] = temperature

    # GPT-5 models require max_completion_tokens instead of max_tokens
    if "gpt-5" in model:
        completion_kwargs["max_completion_tokens"] = max_tokens
    elif model.startswith("ollama/") or model.startswith("ollama_chat/"):
        completion_kwargs["max_tokens"] = min(max_tokens, 4096)
    else:
        # Claude Haiku has a 4096 token limit
        if "claude-3-haiku" in model:
            completion_kwargs["max_tokens"] = min(max_tokens, 4096)
        else:
            completion_kwargs["max_tokens"] = max_tokens

    response = litellm.completion(**completion_kwargs)
    response_text = response['choices'][0]['message']['content']  # pyright: ignore
    new_msg_history.append({"role": "assistant", "content": response['choices'][0]['message']['content']})

    # Convert content to text, compatible with MetaGen API
    new_msg_history = [
        {**msg, "text": msg.pop("content")} if "content" in msg else msg
        for msg in new_msg_history
    ]

    return response_text, new_msg_history, {}


if __name__ == "__main__":
    msg = 'Hello there!'
    # Test the default model (Ollama by default)
    print(f"Testing DEFAULT_MODEL: {DEFAULT_MODEL}")
    try:
        output_msg, msg_history, info = get_response_from_llm(msg, model=DEFAULT_MODEL)
        print(f"OK: {output_msg[:200]}...")
    except Exception as e:
        print(f"FAIL: {str(e)[:300]}")

    # Uncomment to test all models:
    # models = [
    #     ("OLLAMA_LLAMA", OLLAMA_LLAMA),
    #     ("OLLAMA_MISTRAL", OLLAMA_MISTRAL),
    #     ("OLLAMA_CODELLAMA", OLLAMA_CODELLAMA),
    #     ("CLAUDE_MODEL", CLAUDE_MODEL),
    #     ("OPENAI_MODEL", OPENAI_MODEL),
    #     ("MLX_QWEN_OPUS", MLX_QWEN_OPUS),
    #     ("OR_GEMMA4_FREE", OR_GEMMA4_FREE),
    #     ("OR_LLAMA4_FREE", OR_LLAMA4_FREE),
    #     ("OR_QWEN3_FREE", OR_QWEN3_FREE),
    # ]
    # for name, model in models:
    #     print(f"\n{'='*50}")
    #     print(f"Testing {name}: {model}")
    #     print('='*50)
    #     try:
    #         output_msg, msg_history, info = get_response_from_llm(msg, model=model)
    #         print(f"OK: {output_msg[:100]}...")
    #     except Exception as e:
    #         print(f"FAIL: {str(e)[:200]}")

    # Quick OpenRouter test (requires OPENROUTER_API_KEY in .env):
    # if OPENROUTER_API_KEY:
    #     print(f"\nTesting OpenRouter free tier: {OR_GEMMA4_FREE}")
    #     try:
    #         output_msg, _, _ = get_response_from_llm(msg, model=OR_GEMMA4_FREE)
    #         print(f"OK: {output_msg[:200]}...")
    #     except Exception as e:
    #         print(f"FAIL: {str(e)[:300]}")
