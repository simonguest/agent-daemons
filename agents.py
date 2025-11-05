import asyncio
from openai import AsyncOpenAI
import os
from multiprocessing import Queue
from typing import Dict, Any


async def _agent(
    worker_id: str,
    inbox: Queue,
    router_queue: Queue,
    agent_registry: Dict,
    name: str = "Agent with no name",
    model: str = "gpt-4o-mini",
    system_prompt: str = "You are a helpful assistant.",
):
    # Initialize OpenAI client (requires OPENAI_API_KEY env var)
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"))

    # Register this agent with its capabilities
    agent_registry[worker_id] = {
        "type": "agent",
        "model": model,
        "name": name,
        "system_prompt": system_prompt,
        "status": "ready",
    }

    print(f"[{worker_id}] Started with model {model}")

    # Conversation history for this worker
    conversation_history = []

    async def call_llm(user_message: str):
        """Make an async call to OpenAI API"""
        try:
            messages = (
                [
                    {
                        "role": "system",
                        "content": system_prompt or "You are a helpful assistant.",
                    }
                ]
                + conversation_history
                + [{"role": "user", "content": user_message}]
            )

            print(f"[{worker_id}] Calling OpenAI API...")
            response = await client.chat.completions.create(
                model=model, messages=messages, temperature=0.7, max_tokens=500
            )

            result = response.choices[0].message.content

            # Update conversation history
            conversation_history.append({"role": "user", "content": user_message})
            conversation_history.append({"role": "assistant", "content": result})

            # Keep history manageable (last 10 messages)
            if len(conversation_history) > 10:
                conversation_history.pop(0)
                conversation_history.pop(0)

            return result

        except Exception as e:
            return f"Error calling LLM: {str(e)}"

    async def handle_message(msg: Dict[str, Any]):
        print("Message received. Invoking the LLM.")
        print(msg)
        response = await call_llm(msg["content"])
        print(response)
        print("Returning response to the sender")
        message = {
            "from": worker_id,
            "to": msg["from"],
            "type": "chat",
            "content": response,
        }
        router_queue.put(message)

    print(f"[{worker_id}] Entering main loop...")

    while True:
        # Check for incoming messages (non-blocking)
        if not inbox.empty():
            msg = inbox.get()
            # Create task to handle message asynchronously
            asyncio.create_task(handle_message(msg))

        # Small sleep to prevent tight loop
        await asyncio.sleep(0.01)


def agent(
    worker_id: str,
    inbox: Queue,
    router_queue: Queue,
    registry: Dict,
    name: str,
    model: str,
    system_prompt: str,
):
    """Wrapper to run async worker in a process"""
    asyncio.run(
        _agent(worker_id, inbox, router_queue, registry, name, model, system_prompt)
    )
