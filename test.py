import asyncio
import signal

from world import WorldManager


async def test_world():
    # Create a new instance of the world manager
    wm = WorldManager()

    # Start listening for any messages sent from agents
    wm.start_user_queue_monitor()

    # Create a couple of new agents
    simon_UUID = wm.spawn_agent("simon", "gpt-4o-mini", "You help students with their homework. You do not know any other languages.")
    yukiko_UUID = wm.spawn_agent("yukiko", "gpt-4o-mini", "You can translate from English to Japanese.")

    # Send a message to simon
    wm.send_message_to_agent(simon_UUID, "What is 'hello world' in Japanese?")

    # Keep event loop running until signal
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, stop_event.set)

    await stop_event.wait()

    # Shutdown the world
    wm.shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(test_world())
    except KeyboardInterrupt:
        pass