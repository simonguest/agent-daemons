import asyncio
import os
import sys

from openai import AsyncOpenAI
from multiprocessing import Queue
from typing import Dict, Any
from loguru import logger


async def _agent(
    id: str,
    inbox: Queue,
    router_queue: Queue,
    agent_registry: Dict,
    name: str = "Agent with no name",
    model: str = "gpt-4o-mini",
    system_prompt: str = "You are a helpful assistant.",
):
    # Initialize logging client
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<blue>Agent {extra[id]}</blue> | <level>{message}</level>",
    )
    agent_logger = logger.bind(id=id)
    agent_logger.info(f"Starting up with model: {model}")

    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    # Register this agent with its capabilities
    agent_registry[id] = {
        "type": "agent",
        "model": model,
        "name": name,
        "system_prompt": system_prompt,
        "status": "ready",
    }

    conversation_history = []

    async def invoke_llm(user_message: str):
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

            agent_logger.info("Invoking LLM")
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
            agent_logger.error({str(e)})
            return f"Error calling LLM: {str(e)}"

    async def handle_message(message: Dict[str, Any]):
        agent_logger.info(f"Message received from {message['from']}")
        response = await invoke_llm(message["content"])
        agent_logger.info(f"Returning response to {message['from']}")
        message = {
            "from": id,
            "to": message["from"],
            "type": "chat",
            "content": response,
        }
        router_queue.put(message)

    agent_logger.info("Entering main loop")
    while True:
        if not inbox.empty():
            msg = inbox.get()
            asyncio.create_task(handle_message(msg))

        # Small sleep to prevent tight loop
        await asyncio.sleep(0.01)


def agent(
    id: str,
    inbox: Queue,
    router_queue: Queue,
    registry: Dict,
    name: str,
    model: str,
    system_prompt: str,
):
    asyncio.run(_agent(id, inbox, router_queue, registry, name, model, system_prompt))
