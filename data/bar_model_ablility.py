import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 模型名称（最终顺序）
models = [
    "GPT-4o", "Gemini 2.0 Flash", "DeepSeek-V3",
    "GPT-o4-mini", "GPT-o3", "Gemini 2.5 Pro", "DeepSeek-R1"
]

# 数据（根据你提供的表格）
livecode_scores = [29.5, np.nan, 27.2, 75.8, 80.2, 73.6, 73.1]
aime2024_scores = [11.7, 27.5, 25.0, 91.7, 89.2, 87.5, 70.0]

# 横轴：两个数据集
datasets = ["LiveCodeBench (24/8.1–25/5)", "AIME 2024"]
x = np.arange(len(datasets))
width = 0.1  # 每个柱子的宽度

# 颜色映射（淡色，按公司）
color_map = {
    "GPT-4o": "#d9d9d9",
    "GPT-o4-mini": "#a6a6a6",
    "GPT-o3": "#bfbfbf",
    "Gemini 2.0 Flash": "#b0e0dc",
    "Gemini 2.5 Pro": "#9dd8d4",
    "DeepSeek-V3": "#a6c8ff",
    "DeepSeek-R1": "#85b6ff"
}

# 花纹映射（粗线，适合黑白打印）
hatch_map = {
    "GPT-4o": "/",
    "GPT-o4-mini": ".",
    "GPT-o3": "+",
    "Gemini 2.0 Flash": "\\",
    "Gemini 2.5 Pro": "o",
    "DeepSeek-V3": "x",
    "DeepSeek-R1": "*"
}

plt.rcParams["font.family"] = "Times New Roman"

# 自定义图例句柄（大图标，保留花纹）
legend_handles = [
    Patch(facecolor=color_map[model],
          edgecolor='white',
          hatch=hatch_map[model],
          label=model,
          linewidth=1.5)
    for model in models
]

# 绘图
fig, ax = plt.subplots(figsize=(14, 6))

for i, model in enumerate(models):
    color = color_map[model]
    hatch = hatch_map[model]
    scores = [livecode_scores[i], aime2024_scores[i]]
    x_positions = x + (i - len(models) / 2) * width + width / 2
    bars = ax.bar(
        x_positions,
        scores,
        width,
        color=color,
        edgecolor='white',
        hatch=hatch,
        linewidth=1.5
    )
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=26)
        else:
            # For NaN values, show "NA¹" at a fixed height above x-axis
            ax.annotate('N/A',
                        xy=(bar.get_x() + bar.get_width() / 2, 5),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=26,
                        fontweight='bold')

# 设置坐标轴与网格
ax.set_ylabel("Scores (%)", fontsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=28)
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 移除顶部和右侧边框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# 图例（两行，顶部居中，无边框）
ax.legend(
    handles=legend_handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.4),
    ncol=4,
    frameon=False,
    prop={'size': 20, 'family': 'Times New Roman'},
    handlelength=2.5,
    handleheight=1.5
)

plt.tight_layout()

plt.savefig("data/images/bar_model_ablility.png", dpi=300, bbox_inches='tight')
plt.show()
