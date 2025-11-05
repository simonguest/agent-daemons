from multiprocessing import Queue
from typing import Dict
import uuid
import asyncio


class MessageRouter:
    def __init__(
        self, registry, global_queue, agent_queues: Dict[uuid.UUID, Queue], user_queue
    ):
        self.registry = registry
        self.global_queue: Queue = global_queue
        self.agent_queues = agent_queues
        self.user_queue = user_queue

    def queue_message(self, message):
        self.global_queue.put(message)

    async def _route_message(self, message):
        print(f"[Router] Routing message {message}")
        target = message.get("to")

        if target:
            # Check if target is "user"
            if target == "user":
                self.user_queue.put(message)
                print("[Router] Sent message to user")
            else:
                try:
                    # Try to convert target to UUID
                    target_uuid = (
                        uuid.UUID(str(target))
                        if not isinstance(target, uuid.UUID)
                        else target
                    )

                    print(self.agent_queues)
                    # Check if this UUID exists in agent_queues
                    if target_uuid in self.agent_queues:
                        self.agent_queues[target_uuid].put(message)
                        print(f"[Router] Sent message to agent {target_uuid}")
                    else:
                        print(f"[Router] Agent {target_uuid} not found in agent_queues")
                except (ValueError, AttributeError):
                    # Not a valid UUID, could be a broadcast or other routing
                    print(f"[Router] Target '{target}' is not a valid UUID")
        else:
            print("[Router] No target specified in message")

    async def run(self):
        print("[Router] Starting message router...")
        while True:
            if not self.global_queue.empty():
                message = self.global_queue.get()
                asyncio.create_task(self._route_message(message))
            await asyncio.sleep(0.01)
