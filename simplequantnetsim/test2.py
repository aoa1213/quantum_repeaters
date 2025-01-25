import pickle

# 修改为你的文件路径
file_path = "loop_state.pkl"

# 加载数据
with open(file_path, "rb") as f:
    data = pickle.load(f)

# 打印数据内容
print(data)
