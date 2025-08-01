import matplotlib.pyplot as plt
import numpy as np

# 模型和方法
models = [
    "Average", "DeepSeek-V3", "Gemini 2.0 Flash", "GPT-4o", 
    "DeepSeek-R1", "Gemini 2.5 Pro", "GPT-o3"
]
methods = [
    "Direct Code Generation", 
    "Math Agent + Code Agent", 
    "Math Agent + Code\n Agent + Debugging Agent"
]

# 每个模型下对应的 3 个方法得分（移除Single-Step Modeling + Coding）
original_scores = [
    [79.27, 75.61, 75.61],  # 移除第二个值
    [80.49, 73.17, 71.95],
    [82.93, 75.61, 73.17],
    [52.44, 45.12, 40.24],
    [65.85, 62.20, 50.00],
    [69.51, 63.41, 62.20]
]

# 计算每个方法的平均值（现在是3个方法）
averages = []
for i in range(3):  # 3个方法
    method_avg = sum(score[i] for score in original_scores) / len(original_scores)
    averages.append(round(method_avg, 2))

# 将平均值添加到scores中，然后反转顺序
scores = [averages] + original_scores[::-1]

# 配色与填充样式（现在3个方法）
colors = ['lightgray', 'lightblue', 'darkgray']
hatches = ['//', '*', '.']

# 设置字体为 Times New Roman（如果系统支持）
plt.rcParams["font.family"] = "Times New Roman"

# 绘图设置
group_spacing = 2.0  # 增加组间距
y = np.arange(len(models)) * group_spacing
total_width = 1.8  # 增加宽度使条形更粗
bar_width = total_width / len(methods)

fig, ax = plt.subplots(figsize=(14, 12))  # 进一步增加高度以适应更大的间距和条形

# 绘制每种方法对应的柱子（反向绘制以保持图例顺序）
for i, method in enumerate(methods):
    # 使用反向索引来绘制条形，但保持图例的原始顺序
    reverse_i = len(methods) - 1 - i
    method_scores = [scores[j][reverse_i] for j in range(len(models))]
    bars = ax.barh(y + reverse_i * bar_width, method_scores, height=bar_width,
                   label=method, color=colors[i], hatch=hatches[i],
                   edgecolor='white', linewidth=1.5)
    
    # 添加条形内部数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(width - 3, bar.get_y() + bar.get_height() / 2,
                f'{width:.2f}', ha='right', va='center', fontsize=20, 
                color='black', weight='bold')

# 设置坐标轴标签和刻度
ax.set_xlabel("Accuracy (%)", fontsize=40)
ax.set_yticks(y + bar_width * (len(methods) - 1) / 2)  # 自动适应3个方法
ax.set_yticklabels(models, fontsize=40)
ax.tick_params(axis='x', labelsize=40)
ax.set_xlim(0, 90)
ax.set_ylim(-0.5, y[-1] + total_width + 0.5)  # 调整y轴上限以适应更粗的条形
ax.grid(axis='x', linestyle='--', alpha=0.5)

# 移除顶部和右侧边框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加从"Direct Code Generation"到"Math Agent + Code Agent"的弯曲箭头（Average组）
# Average组的y坐标是y[0]，使用与绘图相同的位置计算
avg_group_y = y[0]
# Direct Code Generation: reverse_i=2, 条形中心位置
direct_code_y_center = avg_group_y + 2 * bar_width
# Math Agent + Code Agent: reverse_i=1, 条形中心位置  
math_agent_code_y_center = avg_group_y + 1 * bar_width
# Math Agent + Code Agent + Debugging Agent: reverse_i=0, 条形中心位置
math_agent_debug_y_center = avg_group_y + 0 * bar_width

direct_code_x = 62.20  # Direct Code Generation的值
math_agent_code_x = 66.26  # Math Agent + Code Agent的值
math_agent_debug_x = 71.75  # Math Agent + Code Agent + Debugging Agent的值

# 添加第一个弯曲箭头
ax.annotate('', xy=(math_agent_code_x, math_agent_code_y_center), 
            xytext=(direct_code_x, direct_code_y_center),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', 
                          color='red', lw=2))

# 添加"4.06%"标签 - 移到箭头左侧
arrow_mid_x = direct_code_x + 10
arrow_mid_y = (direct_code_y_center + math_agent_code_y_center) / 2
ax.text(arrow_mid_x, arrow_mid_y, '4.06%', fontsize=22, color='red', 
        weight='bold', ha='center', va='center')

# 添加第二个弯曲箭头
ax.annotate('', xy=(math_agent_debug_x, math_agent_debug_y_center), 
            xytext=(math_agent_code_x, math_agent_code_y_center),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', 
                          color='blue', lw=2))

# 添加"5.49%"标签
arrow2_mid_x = math_agent_code_x + 10
arrow2_mid_y = (math_agent_code_y_center + math_agent_debug_y_center) / 2
ax.text(arrow2_mid_x, arrow2_mid_y, '5.49%', fontsize=22, color='blue', 
        weight='bold', ha='center', va='center')

# 图例布局：第一行两项，第二行左侧一项
handles, labels = ax.get_legend_handles_labels()

# 重新排列图例以确保正确的行布局
# 第一行：handles[0], handles[1]
# 第二行：handles[2]
from matplotlib.patches import Rectangle
blank_handle = Rectangle((0,0), 0, 0, fill=False, edgecolor='none', visible=False)

legend = ax.legend([handles[0], handles[1], blank_handle, handles[2]], 
                   [labels[0], labels[1], ' ', labels[2]], 
                   title="", loc="upper center", bbox_to_anchor=(0.35, 1.12),
                   ncol=3, fontsize=24, handlelength=2.5, columnspacing=0.5, frameon=False)


plt.tight_layout()
plt.savefig("data/images/bar_agent_mode.png", dpi=300, bbox_inches='tight')
plt.show()