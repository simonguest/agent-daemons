from multiprocessing import Process, Manager
from typing import List, Tuple
import uuid
import time
import asyncio

from agents import agent
from message_router import MessageRouter


def _run_message_router(registry, router_queue, agent_queues, user_queue):
    router = MessageRouter(registry, router_queue, agent_queues, user_queue)
    asyncio.run(router.run())


class WorldManager:
    agents = []

    def __init__(self):
        self.manager = Manager()
        self.registry = self.manager.dict()
        self.msg_router_queue = self.manager.Queue()
        self.agent_queues = self.manager.dict()
        self.user_queue = self.manager.Queue()  # Queue for user messages

        # Start message router in separate process
        self.router_process = Process(
            target=_run_message_router,
            args=(
                self.registry,
                self.msg_router_queue,
                self.agent_queues,
                self.user_queue,
            ),
            daemon=True,
        )
        self.router_process.start()

    def start_agent(self, name: str, model: str, system_prompt: str):
        id = uuid.uuid4()
        inbox = self.manager.Queue()

        self.agent_queues[id] = inbox

        p = Process(
            target=agent,
            args=(
                id,
                inbox,
                self.msg_router_queue,
                self.registry,
                name,
                model,
                system_prompt,
            ),
            daemon=True,
        )
        p.start()

        self.agents.append({"id": id, "name": name, "process": p, "inbox": inbox})

        # Give process time to initialize and return id to caller
        time.sleep(2)
        return id

    def list_agents(self) -> List[Tuple]:
        print(self.registry)
        return [(agent["id"], agent["name"]) for agent in self.agents]

    def send_message_to_agent(self, id: uuid.UUID, message: str) -> None:
        new_message = {"from": "user", "to": id, "type": "chat", "content": message}
        self.msg_router_queue.put(new_message)

    async def monitor_user_queue(self) -> None:
        """Monitor the user_queue and print any messages received"""
        print("[WorldManager] Starting user queue monitor...")
        while True:
            if not self.user_queue.empty():
                message = self.user_queue.get()
                content = message.get("content", "")
                sender = message.get("from", "unknown")
                print(f"\n[Message from {sender}]: {content}\n")
            await asyncio.sleep(0.01)

    def start_user_queue_monitor(self) -> None:
        """Start monitoring the user queue in a background asyncio task"""
        try:
            asyncio.get_running_loop()
            asyncio.create_task(self.monitor_user_queue())
        except RuntimeError:
            # No event loop running, need to create one
            print(
                "[WorldManager] Warning: No asyncio event loop running. Call monitor_user_queue() from an async context."
            )

    def stop_agent(self, id: uuid.UUID) -> None:
        for i, a in enumerate(self.agents):
            if a["id"] == id:
                process = a["process"]
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                del self.agents[i]
                if id in self.agent_queues:
                    del self.agent_queues[id]
                break

    def shutdown(self):
        print("Shutting down world")
