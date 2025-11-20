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
            "name": "get_entities",
            "description": "Returns a list of entities that you can communicate with",
            "parameters": {},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Sends a message to another entity",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The id of the entity to send the message to",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content of the message to send",
                    },
                },
                "required": ["id", "content"],
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


def get_entities(registry: DictProxy[Any, Any]):
    """Returns a list of the entities in the world that you can communicate with"""
    return str(registry)


def send_message(router_queue: Queue, from_id: str, to_id: str, content: str):
    """Sends a message to another entity"""
    message = {
        "from": from_id,
        "to": to_id,
        "type": "chat",
        "content": content,
    }
    router_queue.put(message)


available_functions = {
    "get_current_weather": get_current_weather,
    "get_entities": get_entities,
    "send_message": send_message
}
