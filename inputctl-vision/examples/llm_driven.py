#!/usr/bin/env python3
"""
LLM-Driven Pattern Example

In this pattern, the LLM controls the automation flow using tools.
The LLM can see the screen, click, type, and navigate autonomously.

Usage:
    # Set your Anthropic API key
    export ANTHROPIC_API_KEY="your-key-here"

    # Install dependencies
    pip install anthropic

    # Build and install visionctl
    cd visionctl
    maturin develop --uv

    # Run example
    python examples/llm_driven.py

Requirements:
    - KDE Plasma 6.0+ with KWin
    - Anthropic API key
    - /dev/uinput access
    - anthropic Python package
"""

import os
import sys

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed")
    print("Install with: pip install anthropic")
    sys.exit(1)

try:
    from visionctl import VisionCtl
except ImportError:
    print("ERROR: visionctl not installed")
    print("Run: cd visionctl && maturin develop --uv")
    sys.exit(1)


def convert_tools_to_anthropic_format(tools):
    """Convert VisionCtl tool definitions to Anthropic format"""
    anthropic_tools = []
    for tool in tools:
        anthropic_tools.append({
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["input_schema"]
        })
    return anthropic_tools


def main():
    print("=== LLM-Driven GUI Automation Example ===\n")

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    # Create VisionCtl (headless - no LLM backend needed since Claude will handle vision)
    ctl = VisionCtl.new_headless()

    # Get tool definitions for Claude
    tools_raw = ctl.get_tool_definitions()
    tools = convert_tools_to_anthropic_format(tools_raw)

    print(f"Available tools for Claude: {[t['name'] for t in tools]}\n")

    # Create Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    # Initial user request
    user_request = """
    I want you to explore what's on my screen. Please:
    1. Take a screenshot
    2. Describe what you see
    3. If you see a terminal or text editor, click on it

    Be careful and ask before doing anything destructive!
    """

    messages = [
        {"role": "user", "content": user_request}
    ]

    print("Starting LLM-driven automation...\n")
    print("=" * 70)

    max_turns = 10  # Safety limit
    turn = 0

    while turn < max_turns:
        turn += 1
        print(f"\n--- Turn {turn} ---")

        # Call Claude with tools
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            tools=tools,
            messages=messages
        )

        print(f"Stop reason: {response.stop_reason}")

        # Process response
        for block in response.content:
            if block.type == "text":
                print(f"\nClaude: {block.text}")

            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input

                print(f"\nClaude wants to use tool: {tool_name}")
                print(f"Parameters: {tool_input}")

                # Execute the tool
                try:
                    result = ctl.execute_tool(tool_name, tool_input)
                    print(f"Tool result: {result}")

                    # Add tool result to messages
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result)
                        }]
                    })

                except Exception as e:
                    print(f"Tool execution error: {e}")
                    # Report error to Claude
                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error: {str(e)}",
                            "is_error": True
                        }]
                    })

        # If Claude is done (no more tool calls), break
        if response.stop_reason == "end_turn":
            print("\nClaude has finished the task.")
            break

        # If we processed tool uses, continue the loop to get Claude's next response
        if response.stop_reason == "tool_use":
            continue

        # Otherwise, we're done
        break

    print("\n" + "=" * 70)
    print("\n=== Example Complete ===")
    print("\nKey Points:")
    print("- LLM controls the automation flow")
    print("- LLM can see screen via screenshot tool")
    print("- LLM decides which tools to use and when")
    print("- Script just executes tools as requested")
    print("- LLM relies on direct pointing or coordinates for locations")


if __name__ == "__main__":
    main()
