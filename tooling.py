import json

from typing import Any
from multiprocessing import Queue
from multiprocessing.managers import DictProxy
from openai.types.chat import ChatCompletionToolParam

tools: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_agents",
            "description": "Returns a list of AI agents that you can communicate with",
            "parameters": {},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message_to_user",
            "description": "Sends a message to the human user",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content of the message to send",
                    }
                },
                "required": ["location"],
            },
        },
    },
]


def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    weather_data = {
        "location": location,
        "temperature": "72",
        "unit": unit,
        "forecast": ["sunny", "windy"],
    }
    return json.dumps(weather_data)


def get_all_agents(registry: DictProxy[Any, Any]):
    """Returns a list of the current agents that you can communicate with"""
    return str(registry)


def send_message_to_user(router_queue: Queue, id: str, content: str):
    """Sends a message to the human user"""
    message = {
        "from": id,
        "to": "user",
        "type": "chat",
        "content": content,
    }
    router_queue.put(message)


available_functions = {
    "get_current_weather": get_current_weather,
    "get_all_agents": get_all_agents,
    "send_message_to_user": send_message_to_user,
}
