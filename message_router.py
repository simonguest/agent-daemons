import uuid
import sys
import asyncio
from loguru import logger

from multiprocessing import Queue
from typing import Dict


class MessageRouter:
    def __init__(
        self, registry, global_queue, agent_queues: Dict[uuid.UUID, Queue], user_queue
    ):
        logger.remove()
        logger.add(
            sys.stdout,
            colorize=True,
            format="<red>Router</red> | <level>{message}</level>",
            filter=lambda record: record["extra"].get("name") == "router"
        )
        self.router_logger = logger.bind(name="router")
        self.router_logger.info("Starting Message Router")
    
        self.registry = registry
        self.global_queue: Queue = global_queue
        self.agent_queues = agent_queues
        self.user_queue = user_queue

    def queue_message(self, message):
        self.global_queue.put(message)

    async def _route_message(self, message):
        self.router_logger.info(f"Routing message from {message["from"]} to {message["to"]}")
        target = message.get("to")

        if target:
            if target == "user":
                self.user_queue.put(message)
                self.router_logger.info("Sent message to user")
            else:
                try:
                    # Should be an agent - convert target to UUID
                    target_uuid = (
                        uuid.UUID(str(target))
                        if not isinstance(target, uuid.UUID)
                        else target
                    )

                    # Validate UUID exists in agent_queues
                    if target_uuid in self.agent_queues:
                        self.agent_queues[target_uuid].put(message)
                        self.router_logger.info(f"Sent message to agent {target_uuid}")
                    else:
                        self.router_logger.error(f"Agent {target_uuid} not found in agent_queues")
                except (ValueError, AttributeError):
                    self.router_logger.error(f"Target '{target}' is not a valid UUID")
        else:
            self.router_logger.error("No target specified in message")

    async def run(self):
        while True:
            if not self.global_queue.empty():
                message = self.global_queue.get()
                asyncio.create_task(self._route_message(message))
            await asyncio.sleep(0.01)
