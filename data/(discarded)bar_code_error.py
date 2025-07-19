import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 原始数据
data = {
    "Model": ["o3 agent", "o3", "gemini-2.5-pro agent", "gemini-2.5-pro", "DeepSeek-R1 agent", "DeepSeek-R1",
              "gpt-4o agent", "gpt-4o", "gemini-2.0-flash agent", "gemini-2.0-flash", "DeepSeek-V3 agent", "DeepSeek-V3"],
    "IndustryOR": [1.00, 4.00, 0.00, 6.00, 3.00, 13.00, 1.00, 17.00, 2.00, 8.00, 0.00, 12.00],
    "ComplexLP": [0.00, 4.74, 0.00, 3.32, None, 3.32, 0.00, 1.90, 0.00, 3.79, 0.47, 2.37],
    "EasyLP": [0.00, 1.07, 0.00, 1.07, None, 1.38, 0.15, 2.30, 0.00, 0.15, 0.15, 1.07],
    "NL4OPT": [0.00, 0.41, 0.00, 0.41, 0.82, 1.22, 2.45, 6.53, 0.00, 0.00, 0.00, 2.86],
    "BWOR": [1.22, 3.66, 0.00, 2.44, 0.00, 4.88, 2.44, 9.76, 0.00, 4.88, 0.00, 13.41]
}
df = pd.DataFrame(data)
df_melted = df.melt(id_vars="Model", var_name="Dataset", value_name="Error")

# 映射模型名（标准化）
name_map = {
    "gpt-4o": "GPT-4o",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "DeepSeek-V3": "DeepSeek-V3",
    "o3": "GPT-o3",
    "DeepSeek-R1": "DeepSeek-R1",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gpt-4o agent": "GPT-4o agent",
    "gemini-2.0-flash agent": "Gemini 2.0 Flash agent",
    "DeepSeek-V3 agent": "DeepSeek-V3 agent",
    "o3 agent": "GPT-o3 agent",
    "DeepSeek-R1 agent": "DeepSeek-R1 agent",
    "gemini-2.5-pro agent": "Gemini 2.5 Pro agent",
}
df_melted["Model"] = df_melted["Model"].map(name_map)

# 模型与样式定义（灰/蓝/绿 + 不重复花纹）
models = df_melted["Model"].unique()
datasets = df_melted["Dataset"].unique()
x = np.arange(len(datasets))
bar_width = 0.07
offsets = np.linspace(-bar_width * len(models) / 2, bar_width * len(models) / 2, len(models))

# 定义颜色分类
color_map = {
    "GPT-4o": "#d3d3d3",
    "GPT-4o agent": "#a9a9a9",
    "GPT-o3": "#c0c0c0",
    "GPT-o3 agent": "#808080",
    "DeepSeek-R1": "#77aadd",
    "DeepSeek-R1 agent": "#5599cc",
    "DeepSeek-V3": "#3a7ca5",
    "DeepSeek-V3 agent": "#2a5d85",
    "Gemini 2.5 Pro": "#a3d5c3",
    "Gemini 2.5 Pro agent": "#76c7b4",
    "Gemini 2.0 Flash": "#c7e9e9",
    "Gemini 2.0 Flash agent": "#91d9d9",
}
hatches = ['//', '\\\\', 'xx', '++', '**', 'oo', '..', '--', '||', 'OO', '.|', '//']
model_styles = {
    model: {"color": color_map.get(model, "#999999"), "hatch": hatches[i]}
    for i, model in enumerate(models)
}

# 设置字体（如系统支持）
plt.rcParams["font.family"] = "Times New Roman"

# 绘图
fig, ax = plt.subplots(figsize=(16, 7))
for i, model in enumerate(models):
    values = df_melted[df_melted["Model"] == model]["Error"].values
    style = model_styles[model]
    bars = ax.bar(x + offsets[i], values, bar_width,
                  color=style["color"], hatch=style["hatch"],
                  edgecolor='white', label=model)
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2.0, height + 0.3,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

# 格式设定
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylabel("Error (%)", fontsize=12)
ax.set_ylim(0, max(df_melted["Error"].dropna()) + 5)
plt.grid(axis='y', linestyle='--', linewidth=0.5)

# 图例加大字体
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
          ncol=4, fontsize=16, frameon=False)

plt.tight_layout()

plt.savefig("data/images/bar_code_error.png", dpi=300, bbox_inches='tight')
plt.show()