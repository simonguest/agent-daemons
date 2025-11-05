import shlex
import asyncio
import uuid

from world import WorldManager


async def interactive_mode(wm: WorldManager):
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("\nAvailable commands:")
    print("  list                          - List all running agents")
    print(
        "  start <name> <model> <prompt> - Start new agent with name, model, and prompt"
    )
    print("  send <id> <message>           - Send a message to an agent based on id")
    print("  stop <id>                     - Stop agent")
    print("  quit                          - Quit")
    print()

    while True:
        try:
            # Use asyncio to allow other tasks to run while waiting for input
            loop = asyncio.get_event_loop()
            cmd = await loop.run_in_executor(None, input, "> ")
            cmd = cmd.strip()

            if not cmd:
                continue

            elif cmd == "quit":
                break

            elif cmd == "exit":
                break

            elif cmd == "list":
                for agent in wm.list_agents():
                    print(f"{agent[0]} ({agent[1]})")

            elif cmd.startswith("start "):
                parts = cmd.split(" ", 1)[1]
                try:
                    args = shlex.split(parts)
                    if len(args) >= 3:
                        name = args[0]
                        model = args[1]
                        prompt = args[2]
                        agent_id = wm.start_agent(name, model, prompt)
                    else:
                        print("Usage: start <name> <model> <prompt>")
                        print(
                            "Example: start 'Travel Agent' gpt-4o-mini 'You help people book travel'"
                        )
                except ValueError as e:
                    print(f"Error parsing command: {e}")
                    print("Usage: start <name> <model> <prompt>")
                    print(
                        "Example: start 'Travel Agent' gpt-4o-mini 'You help people book travel'"
                    )

            elif cmd.startswith("send "):
                parts = cmd.split(" ", 1)[1]
                try:
                    args = shlex.split(parts)
                    if len(args) >= 2:
                        agent_id = uuid.UUID(args[0])
                        message = args[1]
                        wm.send_message_to_agent(agent_id, message)
                    else:
                        print("Usage: send <id> <message>")
                        print(
                            "Example: send 12345678-1234-5678-1234-567812345678 'Hello agent'"
                        )
                except ValueError as e:
                    print(f"Error parsing command: {e}")
                    print("Usage: send <id> <message>")
                    print(
                        "Example: send 12345678-1234-5678-1234-567812345678 'Hello agent'"
                    )

            elif cmd.startswith("stop "):
                id = cmd.split(" ", 1)[1]
                wm.stop_agent(uuid.UUID(id))

            else:
                print("Unknown command. Type 'quit' to leave interactive mode.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


async def main():
    # Create a new instance of the world manager
    wm = WorldManager()
    wm.start_agent("My first agent", "gpt-4o-mini", "You are a helpful assistant")

    # Start listening for any messages sent from agents
    wm.start_user_queue_monitor()

    # Enter interactive mode
    try:
        await interactive_mode(wm)
    except KeyboardInterrupt:
        pass

    wm.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
