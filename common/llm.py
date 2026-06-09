"""Shared LLM factory for all agents.

Uses OpenRouter as an OpenAI-compatible API, so any provider's model
can be selected via the OPENROUTER_MODEL env var.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client. OpenRouter is primary; falls back to Ollama if no key."""
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()

    if provider == "gemini" or provider == "google":
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            return ChatOpenAI(
                model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
                openai_api_key=gemini_key,
                openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        else:
            # Fallback to local Ollama if key is missing
            return ChatOpenAI(
                model=os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
                openai_api_key="ollama",
                openai_api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1"),
            )

    elif provider == "openrouter":
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            return ChatOpenAI(
                model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
                openai_api_key=openrouter_key,
                openai_api_base="https://openrouter.ai/api/v1",
            )
        else:
            # Fallback to local Ollama if key is missing
            return ChatOpenAI(
                model=os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
                openai_api_key="ollama",
                openai_api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1"),
            )

    elif provider == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                openai_api_key=openai_key,
            )
        else:
            # Fallback to local Ollama if key is missing
            return ChatOpenAI(
                model=os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
                openai_api_key="ollama",
                openai_api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1"),
            )

    else:
        # Default to local Ollama
        return ChatOpenAI(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:latest"),
            openai_api_key="ollama",
            openai_api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1"),
        )