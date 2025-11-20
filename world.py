import uuid
import asyncio
import sys

from multiprocessing import Process, Manager
from typing import List, Tuple
from loguru import logger

from agents import agent
from router import MessageRouter


def _run_message_router(registry, router_queue, agent_queues):
    router = MessageRouter(registry, router_queue, agent_queues)
    asyncio.run(router.run())


class WorldManager:
    def __init__(self):
        logger.remove()
        logger.add(
            sys.stdout,
            colorize=True,
            format="<yellow>World</yellow> | <level>{message}</level>",
            filter=lambda record: record["extra"].get("name") == "world",
        )
        self.world_logger = logger.bind(name="world")
        self.world_logger.info("Launching new world")

        self.manager = Manager()
        self.registry = self.manager.dict()
        self.msg_router_queue = self.manager.Queue()
        self.agent_queues = self.manager.dict()
        self.agent_processes = []  # Instance variable, not class variable
    
        # Start message router in separate process
        self.router_process = Process(
            target=_run_message_router,
            args=(
                self.registry,
                self.msg_router_queue,
                self.agent_queues,
            ),
            daemon=True,
        )
        self.router_process.start()

        # Set the ID of the human interacting with this world and add them to the registry, create them a queue also
        self.user_id = uuid.uuid4()
        self.registry[self.user_id] = {
            "type": "human",
            "name": "Human"
        }
        self.agent_queues[self.user_id] = self.manager.Queue()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()
        # Return False to propagate any exceptions that occurred
        return False

    def spawn_agent(self, name: str, model: str, instructions: str):
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
                instructions,
            ),
            daemon=True,
        )
        p.start()

        self.agent_processes.append({"id": id, "process": p})
        return id

    def list_agents(self) -> List[Tuple[uuid.UUID, str]]:
        return [(agent_id, agent_info['name']) for agent_id, agent_info in self.registry.items()]

    def ping_agent(self, id: uuid.UUID) -> None:
        ping = {"from": self.user_id, "to": id, "type": "ping"}
        self.msg_router_queue.put(ping)

    def request_conversation_history(self, id: uuid.UUID) -> None:
        request = {"from": self.user_id, "to": id, "type": "conversation_history"}
        self.msg_router_queue.put(request)


    def send_message_to_agent(self, id: uuid.UUID, message: str) -> None:
        new_message = {"from": self.user_id, "to": id, "type": "chat", "content": message}
        self.msg_router_queue.put(new_message)

    async def monitor_user_queue(self) -> None:
        logger.add(
            sys.stdout,
            colorize=True,
            format="<green>User</green> | <level>{message}</level>",
            filter=lambda record: record["extra"].get("name") == "user",
        )
        user_logger = logger.bind(name="user")
        while True:
            if not self.agent_queues[self.user_id].empty():
                message = self.agent_queues[self.user_id].get()
                content = message.get("content", "")
                sender = message.get("from", "unknown")
                user_logger.info(f"Message from {sender}: {content}")
            await asyncio.sleep(0.01)

    def start_user_queue_monitor(self) -> None:
        self.world_logger.info("Starting user queue monitor")
        asyncio.get_running_loop()
        asyncio.create_task(self.monitor_user_queue())

    def kill_agent(self, id: uuid.UUID) -> None:
        # Remove from registry
        self.world_logger.info(f"Removing agent {id} from registry")
        del self.registry[id]

        self.world_logger.info(f"Killing agent process for {id}")
        for i, a in enumerate(self.agent_processes):
            if a["id"] == id:
                process = a["process"]
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                del self.agent_processes[i]
                if id in self.agent_queues:
                    del self.agent_queues[id]
                break

    def shutdown(self):
        self.world_logger.info("Shutting down world")

        # Kill all agent processes using kill_agent
        agent_ids = [a["id"] for a in self.agent_processes.copy()]
        for agent_id in agent_ids:
            self.kill_agent(agent_id)

        # Stop the message router process
        if self.router_process.is_alive():
            self.world_logger.info("Stopping message router process")
            self.router_process.terminate()
            self.router_process.join(timeout=5)
            if self.router_process.is_alive():
                self.world_logger.warning("Router process did not terminate, forcing kill")
                self.router_process.kill()

        self.world_logger.info("World shutdown complete")
