import matplotlib.pyplot as plt
import numpy as np

# 模型名称与数据
models = ["GPT-o3", "Gemini 2.5 Pro", "DeepSeek-R1", "GPT-4o", "Gemini 2.0 Flash", "DeepSeek-V3"]
math_error_agent = [80.25, 80.49, 82.93, 53.75, 65.85, 69.51]
math_error_no_agent = [78.48, 73.75, 76.92, 44.59, 52.56, 71.83]

# 设置柱状图参数
x = np.arange(len(models))
width = 0.35

# 设置字体为 Times New Roman（如系统中可用）
plt.rcParams["font.family"] = "Times New Roman"

# 绘图
fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width/2, math_error_agent, width,
               label='OR-LLM-Agent', hatch='//',
               color='lightgray', edgecolor='white')

bars2 = ax.bar(x + width/2, math_error_no_agent, width,
               label='Reasoning and Non-Reasoning LLMs', hatch='\\\\',
               color='lightblue', edgecolor='white')

# 添加柱状图顶部数值标签
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=14)

# 设置标签与图例
ax.set_ylabel('Scores (%)', fontsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=18)
ax.set_ylim(0, 100)
ax.grid(axis='y', linestyle='--', alpha=0.5)

# 移除顶部和右侧边框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# 图例置顶
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=2, fontsize=16, frameon=False)

plt.tight_layout()
plt.savefig("data/images/bar_math_accurary.png", dpi=300, bbox_inches='tight')
plt.show()