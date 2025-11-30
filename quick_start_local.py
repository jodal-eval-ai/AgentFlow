#!/usr/bin/env python3
"""
AgentFlow with Local Model (via vLLM)
======================================

Prerequisites:
1. Start the vLLM server first:
   
   Option A - AgentFlow Planner (port 8000):
       bash scripts/serve_vllm.sh
   
   Option B - Qwen2.5-7B-Instruct (port 8001):
       bash scripts/serve_qwen_local.sh
   
2. Wait for the model to load (check health endpoint)

3. Run this script:
   python quick_start_local.py
"""

from agentflow.agentflow.solver import construct_solver

# ============================================================
# CONFIGURATION - Choose your model setup
# ============================================================

# Option A: Use serve_vllm.sh (AgentFlow Planner on port 8000)
llm_engine_name = "vllm-AgentFlow/agentflow-planner-7b"
base_url = "http://localhost:8000/v1"

# Option B: Use serve_qwen_local.sh (Qwen on port 8001)
# llm_engine_name = "vllm-Qwen/Qwen2.5-7B-Instruct"
# base_url = "http://localhost:8001/v1"

# Construct the solver using local model for ALL agents
solver = construct_solver(
    llm_engine_name=llm_engine_name,
    # Use vLLM for all modules: [planner_main, planner_fixed, verifier, executor]
    model_engine=["trainable", "trainable", "trainable", "trainable"],
    enabled_tools=["Base_Generator_Tool", "Python_Coder_Tool"],
    output_types="final,direct",
    max_steps=5,
    verbose=True,
    base_url=base_url,
    temperature=0.7
)

print("\n" + "="*60)
print("🚀 AgentFlow with Local Qwen2.5-7B-Instruct")
print("="*60)

# Solve a query
output = solver.solve("Calculate the sum of all prime numbers between 1 and 50")

print("\n" + "="*60)
print("✅ Final Answer:")
print("="*60)
print(output.get("direct_output", "No output"))

