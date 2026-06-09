"""End-to-end test client for the Legal Multi-Agent System.

Sends a legal question to the Customer Agent and prints the response.

Uses the new A2A ClientFactory with a long timeout to accommodate
local Ollama LLM inference which can take several minutes per hop.
"""

import asyncio
import os
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

CUSTOMER_AGENT_URL = os.getenv("CUSTOMER_AGENT_URL", "http://localhost:10100")
# Generous timeout: local Ollama needs ~5 min for the full 5-LLM chain
TIMEOUT_SECONDS = int(os.getenv("CLIENT_TIMEOUT", "600"))

QUESTION = (
    "If a company breaks a contract and avoids taxes, "
    "what are the legal and regulatory consequences?"
)


async def main() -> None:
    print(f"Connecting to Customer Agent at {CUSTOMER_AGENT_URL}")
    print(f"Question: {QUESTION}")
    print(f"Timeout: {TIMEOUT_SECONDS}s")
    print("-" * 60)

    timeout = httpx.Timeout(TIMEOUT_SECONDS, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as http_client:
        # Resolve agent card
        card_url = f"{CUSTOMER_AGENT_URL}/.well-known/agent-card.json"
        try:
            card_resp = await http_client.get(card_url)
            card_resp.raise_for_status()
        except Exception:
            # Fallback to deprecated endpoint
            try:
                card_url = f"{CUSTOMER_AGENT_URL}/.well-known/agent.json"
                card_resp = await http_client.get(card_url)
                card_resp.raise_for_status()
            except Exception as e:
                print(f"ERROR: Could not reach Customer Agent at {CUSTOMER_AGENT_URL}")
                print(f"  {e}")
                print("Make sure all services are running (./start_all.sh)")
                sys.exit(1)

        from uuid import uuid4
        from a2a.types import AgentCard, Message, Part, Role, TextPart
        from a2a.types import SendMessageRequest, MessageSendParams
        from a2a.client import A2AClient

        agent_card = AgentCard.model_validate(card_resp.json())
        print(f"Connected to agent: {agent_card.name} v{agent_card.version}")
        print("-" * 60)

        # Use legacy A2AClient but pass the long-timeout httpx client
        client = A2AClient(httpx_client=http_client, agent_card=agent_card)

        message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=QUESTION))],
            message_id=str(uuid4()),
        )
        request = SendMessageRequest(
            id=str(uuid4()),
            params=MessageSendParams(message=message),
        )

        print(f"Sending request (local Ollama may take up to {TIMEOUT_SECONDS//60} min)...\n")
        start = time.time()
        response = await client.send_message(request, http_kwargs={"timeout": TIMEOUT_SECONDS})
        elapsed = time.time() - start
        print(f"[Total latency: {elapsed:.1f}s]\n")

        # Parse response — walk the response tree for text
        result_text = _extract_text(response)

        if result_text:
            print("RESPONSE:")
            print("=" * 60)
            print(result_text)
            print("=" * 60)
        else:
            print("No text response received. Raw response:")
            print(response)


def _extract_text(response: object) -> str:
    """Walk the A2A response tree and collect all TextPart.text values."""
    text = ""
    if hasattr(response, "root"):
        response = response.root
    result = getattr(response, "result", None)
    if result is None:
        return text

    # Task — text lives in artifacts
    artifacts = getattr(result, "artifacts", None)
    if artifacts:
        for artifact in artifacts:
            for part in getattr(artifact, "parts", []) or []:
                inner = getattr(part, "root", part)
                text += getattr(inner, "text", "") or ""
        if text:
            return text

    # Message — text lives in parts directly
    parts = getattr(result, "parts", None)
    if parts:
        for part in parts:
            inner = getattr(part, "root", part)
            text += getattr(inner, "text", "") or ""

    # Fallback: task history messages
    if not text:
        for msg in getattr(result, "history", []) or []:
            for part in getattr(msg, "parts", []) or []:
                inner = getattr(part, "root", part)
                text += getattr(inner, "text", "") or ""

    return text


if __name__ == "__main__":
    asyncio.run(main())