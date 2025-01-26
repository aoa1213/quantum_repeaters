# config.py（位于 simplequantnetsim 文件夹内）
from pathlib import Path

# 项目根目录：假设 simplequantnetsim 是项目根目录的直接子文件夹
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 指向 QUANTUM_REPEATERS_TEST

DATA_PATHS = {
    "input_graphs": PROJECT_ROOT / "graphs_json",          # 输入数据目录
    "output_results": PROJECT_ROOT / "new_result",         # 输出结果目录
    "communities": PROJECT_ROOT / "communities"            # 社区划分结果目录
}