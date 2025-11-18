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
You are a helpful AI Agent.

Your goal is to handle incoming requests, create a plan for how to complete them (if they are complex), and communicate accordingly.

You have access to a set of tools:

get_all_agents: This lets you see other agents who you can communicate with.
send_message_to_agent: This lets you send a message to another agent.
send_message_to_user: This lets you send a message to the user (a human)

Note: The list of agents is dynamic and can change frequently. You should always call get_all_agents before send_message_to_agent.

If you want to send a message back to the agent/use, you must use the send_message_to_agent or send_message_to_user tools. Just adding a reply to the conversation thread isn't enough.

When you send a message to an agent or the user, you'll pause working on your task. If the agent/user sends a follow up message, your task will continue.

You will continue to work on your task until you send a message.

If you received a message from an agent, you should reply back to the agent (using send_message_to_agent) vs. communicating directly with the user (using send_message_to_user).

Here are additional instructions that you should follow:


"""

MAX_THINKING_TURNS = 10 # The maximum number of turns that can be taken before sending a message to prevent inf. loops

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

    async def invoke_llm(conversation_id: str, content: str):
        agent_logger.info("Invoking LLM")
        try:
            task_wip = True # Agent is still working on the task

            # Add the incoming message to the conversation thread
            if content != "":
                update_conversation(
                    conversation_id, {"role": "user", "content": content}
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

                    if function_name == "get_all_agents":
                        function_response = function_to_call(registry=agent_registry)

                    if function_name == "send_message_to_user":
                        function_response = function_to_call(
                            router_queue=router_queue,
                            from_id=id,
                            content=function_args.get("content"),
                        )
                        task_wip = False # Task completed, waiting on user

                    if function_name == "send_message_to_agent":
                        function_response = function_to_call(
                            router_queue=router_queue,
                            from_id=id,
                            to_id=function_args.get("id"),
                            content=function_args.get("content"),
                        )
                        task_wip = False # Task completed, waiting on agent

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

                    if task_wip:
                        # Add tool call response back to the conversation thread
                        response = await client.chat.completions.create(
                            model=model, messages=get_conversation(conversation_id)
                        )
                        response = response.choices[0].message.content
                        update_conversation(
                            conversation_id, {"role": "assistant", "content": response}
                        )

                        # Recursive call to continue the task
                        await invoke_llm(conversation_id, "")
            else:
                if task_wip:
                    # The agent is thinking to itself - add to the conversation thread
                    response = response_message.content
                    update_conversation(
                        conversation_id, {"role": "assistant", "content": response}
                    )

                    # Recursive call to continue the task
                    await invoke_llm(conversation_id, "")

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
            agent_logger.info(f"Chat message received from {message['from']}")
            response = await invoke_llm(message["from"], message["content"])


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
