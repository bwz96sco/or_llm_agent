import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 数据准备
datasets = ['IndustryOR', 'ComplexLP', 'EasyLP', 'NL4OPT', 'BWOR']
models = [
    'GPT-o3 - GPT-4o',
    'GPT-o4-mini - GPT-4o',
    'Gemini 2.5 Pro - Gemini 2.0 Flash',
    'DeepSeek-R1 - DeepSeek-V3'
]
data = [
    [1.00, 3.32, -14.11, 5.31, 35.37],
    [-5.00, 3.32, -1.84, 6.53, 32.93],
    [3.00, -7.11, -17.79, -4.90, 21.95],
    [-1.00, 5.69, -8.28, -1.63, 10.98]
]
df = pd.DataFrame(data, index=models, columns=datasets)

# 柱状图参数
x = np.arange(len(datasets))
bar_width = 0.2

# 样式设定
colors = ['lightgray', 'dimgray', 'powderblue', 'cornflowerblue']
hatches = ['///', '...', 'oo', '**']

# 设置字体为 Times New Roman（如可用）
plt.rcParams['font.family'] = 'Times New Roman'

# 绘图
fig, ax = plt.subplots(figsize=(14, 6))

for i, (model, color, hatch) in enumerate(zip(models, colors, hatches)):
    bars = ax.bar(
        x + i * bar_width,
        df.loc[model],
        width=bar_width,
        label=model,
        color=color,
        hatch=hatch,
        edgecolor='white'  # 使用白色边框
    )
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5 if height >= 0 else height - 2,
            f'{height:.2f}',  # 保留两位小数
            ha='center',
            va='bottom' if height >= 0 else 'top',
            fontsize=18
        )

# 图表美化
ax.set_ylabel('Performance Difference (%)', fontsize=28)
ax.tick_params(axis='y', labelsize=28)
ax.set_xticks(x + 1.5 * bar_width)
ax.set_xticklabels(datasets, fontsize=28)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylim(-25, 40)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, 1.35),
          ncol=2, fontsize=28, handlelength=2.5, columnspacing=1.5, frameon=False)

# 移除顶部和右侧边框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.tight_layout()
plt.savefig("data/images/bar_model_compare.png", dpi=300, bbox_inches='tight')
plt.show()