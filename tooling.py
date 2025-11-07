import json

from typing import Any
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
            "parameters": {
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


available_functions = {
    "get_current_weather": get_current_weather,
    "get_all_agents": get_all_agents
}
