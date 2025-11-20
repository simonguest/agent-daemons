import asyncio
import os
import sys
import json
from datetime import datetime

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCall
from multiprocessing import Queue
from typing import Dict, Any, List, cast
from loguru import logger

from tooling import tools, available_functions

SYSTEM_PROMPT = """
You are a helpful AI Agent operating in a world with many other entities. These other entites can be other AI agents or humans.

Your goal is to handle incoming requests from other entites, complete them, and communicate accordingly.

You have access to a set of tools:

**get_entities**: This lets you see other entities who you can communicate with.

Entities can be humans or other AI agents, similar to yourself. Every entity has a unique ID for communication.

**send_message**: This lets you send a message to another entity.

Note: The list of entities in the world is dynamic and can change frequently. You should always call get_all_entities before sending a message.

When you send a message to an entity, you'll pause working on your task. If the entity sends a follow up message, your task will continue.

You will continue to work on your task until you send a message.

Here are additional instructions that you should follow:

"""

MAX_THINKING_TURNS = 10  # The maximum number of turns that can be taken before sending a message to prevent inf. loops


async def _agent(
    id: str,
    inbox: Queue,
    router_queue: Queue,
    agent_registry: Dict,
    name: str = "Agent with no name",
    model: str = "gpt-4o-mini",
    instructions: str = "",
):
    # Initialize logging client
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<blue>Agent {extra[name]} ({extra[id]})</blue> | <level>{message}</level>",
    )
    agent_logger = logger.bind(name=name, id=id)
    agent_logger.info(f"Starting up with model: {model}")

    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    # Register this agent with its capabilities
    agent_registry[id] = {
        "type": "agent",
        "model": model,
        "name": name,
        "instructions": instructions,
        "status": "ready",
    }

    # Dictionary of conversations for everyone the agent is chatting with
    conversations: Dict[str, List[ChatCompletionMessageParam]] = {}

    def get_conversation(id: str) -> List[ChatCompletionMessageParam]:
        """Get a conversation by ID. Creates a new conversation with system message if it doesn't exist."""
        if id not in conversations:
            conversations[id] = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT + "\n\n" + instructions,
                }
            ]
        return conversations[id]

    def update_conversation(id: str, message: ChatCompletionMessageParam) -> None:
        """Update a conversation by adding a message. Creates the conversation if it doesn't exist."""
        if id not in conversations:
            conversations[id] = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT + "\n\n" + instructions,
                }
            ]
        conversations[id].append(message)

    async def invoke_llm(thinking_turn: int, conversation_id: str, content: str):
        agent_logger.info(f"Thinking... {thinking_turn}")
        if thinking_turn > MAX_THINKING_TURNS:
            agent_logger.info("Reached max number of thinking turns. Giving up.")
            return
        
        try:
            task_wip = True  # Agent is still working on the task

            # Add the incoming message to the conversation thread
            if content != "":
                content_template = f"You have recieved a message from entity: {conversation_id}. The message is {content}"
                update_conversation(
                    conversation_id, {"role": "user", "content": content_template}
                )

            response = await client.chat.completions.create(
                model=model,
                messages=get_conversation(conversation_id),
                temperature=0.7,
                max_tokens=500,
                tools=tools,
            )

            response_message = response.choices[0].message

            # Check to see if a tool call is needed
            tool_calls = response_message.tool_calls
            if tool_calls:
                # Append user message and assistant response with tool calls
                update_conversation(
                    conversation_id, {"role": "user", "content": content}
                )
                update_conversation(
                    conversation_id,
                    cast(ChatCompletionMessageParam, response_message.model_dump()),
                )

                tool_called = False

                for tool_call in tool_calls:
                    # Cast to proper tool call type to access function attribute
                    typed_tool_call = cast(ChatCompletionMessageToolCall, tool_call)
                    function_name = typed_tool_call.function.name
                    agent_logger.info(f"Calling tool: {function_name}")
                    function_to_call = available_functions[function_name]
                    function_args = json.loads(typed_tool_call.function.arguments)

                    if function_name == "get_current_weather":
                        function_response = function_to_call(
                            location=function_args.get("location"),
                            unit=function_args.get("unit", "fahrenheit"),
                        )
                        tool_called = True

                    if function_name == "get_entities":
                        function_response = function_to_call(registry=agent_registry)
                        tool_called = True

                    if function_name == "send_message":
                        function_response = function_to_call(
                            router_queue=router_queue,
                            from_id=id,
                            to_id=function_args.get("id"),
                            content=function_args.get("content"),
                        )
                        tool_called = True
                        task_wip = False  # Task completed, waiting on agent

                    if tool_called:
                        update_conversation(
                            conversation_id,
                            cast(
                                ChatCompletionMessageParam,
                                {
                                    "tool_call_id": typed_tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": function_response or "None",
                                },
                            ),
                        )
                        # Add tool call response back to the conversation thread
                        response = await client.chat.completions.create(
                            model=model, messages=get_conversation(conversation_id)
                        )
                        response = response.choices[0].message.content
                        update_conversation(
                            conversation_id, {"role": "assistant", "content": response}
                        )

                        if task_wip:
                            # Recursive call to continue the task
                            await invoke_llm(thinking_turn+1, conversation_id, "")
                    else:
                        agent_logger.info(f"No tool available: {function_name}")
                        update_conversation(
                            conversation_id, {"role": "assistant", "content": f"I tried to call a tool called {function_name}, but it was not available"}
                        )
                        await invoke_llm(thinking_turn+1, conversation_id, "")
            else:
                if task_wip:
                    # The agent is thinking to itself - add to the conversation thread
                    response = response_message.content
                    update_conversation(
                        conversation_id, {"role": "assistant", "content": response}
                    )

                    # Recursive call to continue the task
                    await invoke_llm(thinking_turn+1, conversation_id, "")

        except Exception as e:
            agent_logger.error(f"Error: {e}")
            return f"Error calling LLM: {str(e)}"

    async def handle_message(message: Dict[str, Any]):
        # Check for message type
        if message["type"] == "ping":  # Check if the agent is alive
            agent_logger.info(f"Ping received from {message['from']}")
            # Create ack message
            ack = {
                "from": id,
                "to": message["from"],
                "type": "ack",
                "content": f"Ack at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            }
            router_queue.put(ack)
        elif (
            message["type"] == "conversation_history"
        ):  # Request for the conversation history
            agent_logger.info(
                f"Received request for conversation history from {message['from']}"
            )
            # Create response
            response = {
                "from": id,
                "to": message["from"],
                "type": "chat",
                "content": get_conversation(message["from"]),
            }
            router_queue.put(response)
        elif message["type"] == "chat":  # A chat message for the agent
            agent_logger.info(
                f"Chat message received from {message['from']}: {message['content']}"
            )
            response = await invoke_llm(1, message["from"], message["content"])

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
    agent_registry: Dict,
    name: str,
    model: str,
    instructions: str,
):
    asyncio.run(
        _agent(id, inbox, router_queue, agent_registry, name, model, instructions)
    )
