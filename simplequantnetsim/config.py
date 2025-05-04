from pathlib import Path

# Project root directory: assuming simplequantnetsim is a direct subfolder of the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Points to QUANTUM_REPEATERS_TEST

DATA_PATHS = {
    "input_graphs": PROJECT_ROOT / "graphs_json",        # Directory for input graph data
    "output_results": PROJECT_ROOT / "new_result",      # Directory for output results
    "communities": PROJECT_ROOT / "communities"          # Directory for community detection results
}
