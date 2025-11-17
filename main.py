import shlex
import asyncio
import uuid

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import HTML

from world import WorldManager


class AgentCompleter(Completer):
    def __init__(self, wm: WorldManager):
        self.wm = wm
        self.commands = ['list', 'spawn', 'send', 'kill', 'quit', 'exit', 'help']

    def get_completions(self, document: Document, _):
        text = document.text_before_cursor
        words = text.split()

        # If we're at the start or completing the first word, suggest commands
        if len(words) == 0 or (len(words) == 1 and not text.endswith(' ')):
            word = words[0] if words else ''
            for cmd in self.commands:
                if cmd.startswith(word.lower()):
                    yield Completion(cmd, start_position=-len(word))

        # If we typed "ping ", "send ", or "kill ", suggest agent IDs
        elif len(words) >= 1 and words[0] in ('ping', 'send', 'conversation', 'kill'):
            if len(words) == 1 or (len(words) == 2 and not text.endswith(' ')):
                # We're completing the agent ID
                current_word = words[1] if len(words) == 2 else ''
                agents = self.wm.list_agents()
                for agent_id, agent_name in agents:
                    agent_id_str = str(agent_id)
                    if agent_id_str.startswith(current_word):
                        # Show agent name as metadata for easier identification
                        yield Completion(
                            agent_id_str,
                            start_position=-len(current_word),
                            display_meta=agent_name
                        )


def print_help():
    print("\nAvailable commands:")
    print("  list                                   - List all running agents")
    print("  spawn <name> <model> <instructions>    - Spawn new agent with name, model, and instructions")
    print("  ping <id>                              - Ping the agent")
    print("  send <id> <message>                    - Send a message to an agent based on id")
    print("  conversation <id>                      - Reveal the conversation history for the agent")
    print("  kill <id>                              - Kill agent")
    print("  help                                   - Show this help message")
    print("  quit/exit                              - Quit")
    print()


async def interactive_mode(wm: WorldManager):
    # Create a custom completer that knows about commands and agent IDs
    completer = AgentCompleter(wm)

    # Create a prompt session
    session = PromptSession(
        message=HTML('<b>&gt;</b> '),
        completer=completer,
    )

    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print_help()
    print("Press Tab for command completion, Ctrl+C to cancel input")
    print()

    while True:
        try:
            # Get input using prompt_toolkit (async-friendly)
            cmd = await session.prompt_async()
            cmd = cmd.strip()

            if not cmd:
                continue

            elif cmd in ("quit", "exit"):
                break

            elif cmd == "help":
                print_help()

            elif cmd == "list":
                agents = wm.list_agents()
                if not agents:
                    print("No agents running.")
                else:
                    for agent in agents:
                        print(f"  {agent[0]} ({agent[1]})")

            elif cmd.startswith("spawn "):
                parts = cmd.split(" ", 1)[1]
                try:
                    args = shlex.split(parts)
                    if len(args) >= 3:
                        name = args[0]
                        model = args[1]
                        instructions = args[2]
                        agent_id = wm.spawn_agent(name, model, instructions)
                        print(f"Spawned agent: {name} (ID: {agent_id})")
                    else:
                        print("Usage: spawn <name> <model> <instructions>")
                        print("Example: spawn 'Travel Agent' gpt-4o-mini 'You help people book travel'")
                except ValueError as e:
                    print(f"Error parsing command: {e}")
                    print("Usage: spawn <name> <model> <instructions>")
                    print("Example: spawn 'Travel Agent' gpt-4o-mini 'You help people book travel'")

            elif cmd.startswith("ping "):
                parts = cmd.split(" ", 1)[1]
                try:
                    args = shlex.split(parts)
                    if len(args) >= 1:
                        agent_id = uuid.UUID(args[0])
                        wm.ping_agent(agent_id)
                        print(f"Pinged agent {agent_id}")
                    else:
                        print("Usage: ping <id>")
                        print("Example: ping 12345678-1234-5678-1234-567812345678")
                except ValueError as e:
                    print(f"Error parsing command: {e}")
                    print("Usage: ping <id>")
                    print("Example: ping 12345678-1234-5678-1234-567812345678")               

            elif cmd.startswith("conversation "):
                parts = cmd.split(" ", 1)[1]
                try:
                    args = shlex.split(parts)
                    if len(args) >= 1:
                        agent_id = uuid.UUID(args[0])
                        wm.request_conversation_history(agent_id)
                    else:
                        print("Usage: conversation <id>")
                        print("Example: conversation 12345678-1234-5678-1234-567812345678")
                except ValueError as e:
                    print(f"Error parsing command: {e}")
                    print("Usage: conversation <id>")
                    print("Example: conversation 12345678-1234-5678-1234-567812345678")        

            elif cmd.startswith("send "):
                parts = cmd.split(" ", 1)[1]
                try:
                    args = shlex.split(parts)
                    if len(args) >= 2:
                        agent_id = uuid.UUID(args[0])
                        message = args[1]
                        wm.send_message_to_agent(agent_id, message)
                        print(f"Message sent to agent {agent_id}")
                    else:
                        print("Usage: send <id> <message>")
                        print("Example: send 12345678-1234-5678-1234-567812345678 'Hello agent'")
                except ValueError as e:
                    print(f"Error parsing command: {e}")
                    print("Usage: send <id> <message>")
                    print("Example: send 12345678-1234-5678-1234-567812345678 'Hello agent'")

            elif cmd.startswith("kill "):
                id_str = cmd.split(" ", 1)[1]
                try:
                    agent_id = uuid.UUID(id_str)
                    wm.kill_agent(agent_id)
                    print(f"Killed agent {agent_id}")
                except ValueError as e:
                    print(f"Error: Invalid UUID format: {e}")

            else:
                print(f"Unknown command: '{cmd}'. Type 'help' for available commands.")

        except KeyboardInterrupt:
            # Ctrl+C cancels current input but doesn't exit
            print("\n(Press Ctrl+D or type 'quit' to exit)")
            continue
        except EOFError:
            # Ctrl+D exits
            break
        except Exception as e:
            print(f"Error: {e}")


async def main():
    # Create a new instance of the world manager
    wm = WorldManager()
    
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
