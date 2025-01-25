

# ============== 2) 全局参数 ==============
funcs = [MPC_protocol, MPG_protocol, SP_protocol]
p_range = np.linspace(1, 0.2, 50)

timesteps = 10
reps = 20
alpha = 1.4
beta = 0.105
max_rounds = 10
shuffle_nodes = True

root_path = Path.cwd().parent.joinpath("graphs_json")

sr_results = []  # 每个文件的 SR 结果都会追加到这里

# 分段退出：针对“组合”数量
chunk_size = 300  
state_file = "loop_state.pkl"

# ============== 3) 读取/初始化进度 ==============
try:
    with open(state_file, "rb") as f:
        progress = pickle.load(f)
    print("恢复进度：", progress)
except FileNotFoundError:
    progress = {
        "subfolder_idx": 0,   # 当前处理到第几个子文件夹
        "file_idx": 0,        # 当前子文件夹内处理到第几个文件
        "combo_idx": 0,       # 当前文件内处理到第几个组合
        "global_combo_count": 0  # 全局已处理的组合数
    }
    print("未发现进度文件，从头开始。")

# 子文件夹列表
subfolders = [sf for sf in root_path.iterdir() if sf.is_dir()]
subfolders.sort()

# ============== 4) 主循环 ==============
for s_idx in range(progress["subfolder_idx"], len(subfolders)):
    subfolder = subfolders[s_idx]
    if not subfolder.is_dir():
        continue

    print(f"Processing subfolder: {subfolder}")

    # 遍历文件
    files = [f for f in subfolder.iterdir() if f.is_file()]
    files.sort()

    for f_idx in range(progress["file_idx"], len(files)):
        file = files[f_idx]
        if not file.is_file():
            continue

        print(f"  Processing file: {file} ...")

        # 每个文件：初始化 counters
        failure_counts = {func.__name__: 0 for func in funcs}
        combination_counter = 0

        # 加载图
        G, users, pos = load_data(file)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray")
        plt.close()

        G = network(G)

        # 输出路径
        class_folder = subfolder.name
        file_path = file.with_suffix(".png")
        file_name = file_path.name

        er_folder_path = Path.cwd().parent.joinpath("new_result", class_folder)
        er_folder_path.mkdir(exist_ok=True)
        er_topology_folder_path = er_folder_path.joinpath(file_name)
        er_topology_folder_path.mkdir(exist_ok=True)
        communities_output_path = Path.cwd().parent.joinpath("communitie", class_folder, file_name)

        communities, users_node_combination = multi_iterative_score_partition_with_drawing(
            G, users[0], alpha, beta, max_rounds, shuffle_nodes, pos, communities_output_path
        )

        # 遍历本文件所有组合
        for combo_idx in range(progress["combo_idx"], len(users_node_combination)):
            combo = users_node_combination[combo_idx]
            combination_counter += 1
            progress["global_combo_count"] += 1

            # 计算 ER
            ER = np.zeros((len(funcs), len(p_range)))
            for i, p in enumerate(tqdm.tqdm(p_range, desc=f"Combo {combo_idx}/{len(users_node_combination)}")):
                set_p_edge(G, p_op=p)
                for j, function in enumerate(funcs):
                    er, mt_gen_time, links_used = function(G, users, timesteps=timesteps, reps=reps)
                    ER[j, i] += er

            # 判断是否全0
            all_zeros = np.all(ER == 0, axis=1)
            for idxp, is_zero in enumerate(all_zeros):
                if is_zero:
                    protocol_name = funcs[idxp].__name__
                    failure_counts[protocol_name] += 1

            # 绘图保存
            er_output_path = er_topology_folder_path.joinpath(f'result_{combo_idx}.png')
            plot_er_vs_p(p_range, ER, funcs, cols, er_output_path)

            # 分段退出：每处理 chunk_size 个组合后退出
            if progress["global_combo_count"] % chunk_size == 0:
                print(f"已处理 {progress['global_combo_count']} 个组合，准备退出。")
                progress["subfolder_idx"] = s_idx
                progress["file_idx"] = f_idx
                progress["combo_idx"] = combo_idx + 1  # 下次从下一个组合继续
                with open(state_file, "wb") as pf:
                    pickle.dump(progress, pf)
                exit()

            # 每个组合处理完更新进度
            progress["combo_idx"] = combo_idx + 1
            progress["subfolder_idx"] = s_idx
            progress["file_idx"] = f_idx
            with open(state_file, "wb") as pf:
                pickle.dump(progress, pf)

        # ===========【重点】在文件的所有组合结束后，计算该文件的SR===========
        sr_for_protocols = {}
        if combination_counter > 0:
            for protocol_name, failures in failure_counts.items():
                sr_for_protocols[protocol_name] = (combination_counter - failures) / combination_counter
        else:
            for protocol_name in failure_counts:
                sr_for_protocols[protocol_name] = 0

        # 添加元信息（子文件夹、文件名）
        sr_for_protocols["Subfolder"] = subfolder.name
        sr_for_protocols["File"] = file.name
        sr_results.append(sr_for_protocols)

        # 这个文件处理完 => 重置 combo_idx，并 file_idx+1
        progress["combo_idx"] = 0
        progress["file_idx"] = f_idx + 1
        with open(state_file, "wb") as pf:
            pickle.dump(progress, pf)

    # 子文件夹处理完成 => 重置 file_idx + subfolder_idx+1
    progress["file_idx"] = 0
    progress["subfolder_idx"] = s_idx + 1
    with open(state_file, "wb") as pf:
        pickle.dump(progress, pf)

# ============== 5) 若所有子文件夹都处理完 => 写出 CSV & 删除进度文件 ==============
if progress["subfolder_idx"] >= len(subfolders):
    print("所有子文件夹都处理完毕，开始写入 CSV。")

    output_csv_path = Path.cwd().parent.joinpath("protocol_sr_results.csv")
    with open(output_csv_path, mode="w", newline="") as csvfile:
        fieldnames = ["Subfolder", "File"] + [func.__name__ for func in funcs]
        csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(sr_results)

    print(f"协议 SR 结果已保存到 {output_csv_path}")
    Path(state_file).unlink(missing_ok=True)
