import matplotlib.pyplot as plt
import numpy as np

# 模型和方法
models = [
    "GPT-o3", "Gemini 2.5 Pro", "DeepSeek-R1", 
    "GPT-4o", "Gemini 2.0 Flash", "DeepSeek-V3"
]
methods = [
    "Code Agent", 
    "Math Agent + Code Agent (in 1 step)", 
    "Math Agent + Code Agent", 
    "Math Agent + Code Agent + Debugging Agent"
]

# 每个模型下对应的 4 个方法得分
scores = [
    [75.61, 75.61, 75.61, 79.27],
    [71.95, 73.17, 75.61, 80.49],
    [73.17, 71.95, 75.61, 82.93],
    [40.24, 39.02, 45.12, 52.44],
    [50.00, 59.76, 62.20, 65.85],
    [62.20, 58.54, 63.41, 69.51]
]

# 配色与填充样式
colors = ['lightgray', 'paleturquoise', 'lightblue', 'darkgray']
hatches = ['//', 'o', '*', '.']

# 设置字体为 Times New Roman（如果系统支持）
plt.rcParams["font.family"] = "Times New Roman"

# 绘图设置
x = np.arange(len(models))
total_width = 0.8
bar_width = total_width / len(methods)

fig, ax = plt.subplots(figsize=(14, 6))

# 绘制每种方法对应的柱子
for i, method in enumerate(methods):
    method_scores = [scores[j][i] for j in range(len(models))]
    bars = ax.bar(x + i * bar_width, method_scores, width=bar_width,
                  label=method, color=colors[i], hatch=hatches[i],
                  edgecolor='white', linewidth=1.5)
    
    # 添加顶部数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12)

# 设置坐标轴标签和刻度
ax.set_ylabel("Scores (%)", fontsize=16)
ax.set_xticks(x + bar_width * (len(methods) - 1) / 2)
ax.set_xticklabels(models, rotation=15, fontsize=18)
ax.tick_params(axis='y', labelsize=16)
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 移除顶部和右侧边框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例放在顶部，两行排列
ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, 1.18),
          ncol=2, fontsize=16, handlelength=2.5, columnspacing=1.5, frameon=False)

plt.tight_layout()
plt.savefig("data/images/bar_agent_mode.png", dpi=300, bbox_inches='tight')
plt.show()