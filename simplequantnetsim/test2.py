from pathlib import Path
import json

def process_graphs_json_folder(root_folder):
    """
    遍历 graphs_json 文件夹中的每个子文件夹及其文件，并对每个文件进行处理。

    参数:
        root_folder (str): graphs_json 文件夹路径
    """
    root_path = Path(root_folder)

    # 遍历 graphs_json 文件夹中的每个子文件夹
    for subfolder in root_path.iterdir():
        if subfolder.is_dir():  # 判断是否是子文件夹
            print(f"Processing subfolder: {subfolder}")
            
            # 遍历当前子文件夹中的所有文件
            for file in subfolder.iterdir():
                if file.is_file():  # 判断是否是文件
                    #print(f"  Processing file: {file}")
                    with open(file, "r") as f:
                        data = json.load(f)

# 示例调用
main_folder = Path(__file__).parent.parent.joinpath("graphs_json")
process_graphs_json_folder(main_folder)