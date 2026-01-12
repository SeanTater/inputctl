import os
import json
from visionctl import VisionCtl

# --- MOCK BRAIN (The Capable Model) ---
# In a real scenario, this would be Claude 3.5 or GPT-4o
def brain_agent():
    # 1. Initialize VisionCtl 
    # Option A: Headless (if the brain only calls tools)
    ctl = VisionCtl.new_headless()
    
    # Configure the 'Eyes' (Tiny Qwen model via Ollama)
    # The brain re-uses the eyes specifically for the 'point_at' tool.
    ctl.set_pointing_model(
        backend="ollama",
        url="http://localhost:11434",
        model="qwen2-vl:2b" # Specialist model
    )
    
    # 2. Get tool definitions to show the brain what it can do
    tools = ctl.get_tool_definitions()
    print("Agent tools available:", [t['name'] for t in tools])

    # 3. Simulate the brain deciding to use 'point_at'
    print("\n[Brain]: I need to click the search bar. I'll ask my vision specialist...")
    
    # Tool call params - description is high level
    params = {
        "description": "the search bar at the top of the browser"
    }
    
    # 4. Execute the tool
    # Internally, VisionCtl captures a screenshot and calls the pointing model.
    # The pointing model returns a JSON bbox, which visionctl parses to move the mouse.
    result = ctl.execute_tool("point_at", params)
    
    print(f"\n[Tool Result]: {result}")
    
    if result.get("success"):
        print("[Brain]: Great! Now I can click.")
        ctl.execute_tool("click", {"button": "left"})
    else:
        print("[Brain]: Vision specialist failed. Maybe I'll try the grid overlay instead.")

# --- MOCK VISION SPECIALIST ---
# To actually run this, you would need Ollama running with qwen2-vl:2b
# curl -fsSL https://ollama.com/install.sh | sh
# ollama run qwen2-vl:2b

if __name__ == "__main__":
    print("--- Pointing Delegation Demo ---")
    try:
        brain_agent()
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This demo requires the visionctl Rust library to be built and installed.")
        print("Run: maturin develop --uv")
