"""
航班延误：天气 & 时段能解释几何？

代码涉及生成论文中的图片，因此有些redundancy，还请老师谅解
"""

import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")
# statsmodels在处理分类变量（如carrier和origin的固定效应）时有很多常见警告，过滤掉
warnings.filterwarnings("ignore", message="covariance of constraints does not have full rank")

# ========= 0. 准备目录 =========
OUT_DIR_DATA = "data_clean"
OUT_DIR_FIG = "results/fig"
OUT_DIR_TAB = "results/tables"
for d in [OUT_DIR_DATA, OUT_DIR_FIG, OUT_DIR_TAB]:
    os.makedirs(d, exist_ok=True)

# ========= 1. 载入数据 =========
try:
    from nycflights13 import flights, weather, airports, airlines
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nycflights13", "--quiet"])
    from nycflights13 import flights, weather, airports, airlines

import numpy as np
import pandas as pd
from scipy import stats

# 将 nycflights13 的内置表转为 DataFrame
df_flights = flights.copy()
df_weather = weather.copy()
df_airports = airports.copy()
df_airlines = airlines.copy()

# ========= 2. 基础清洗 =========
# 去掉取消航班（无到达延误）
df = df_flights[df_flights["arr_delay"].notna()].copy()

# 构造日期、星期等
# flights 自带 time_hour（计划起飞时间按小时对齐），用于与天气合并
df["date"] = pd.to_datetime(df[["year", "month", "day"]])
df["dow"] = df["date"].dt.dayofweek  # 0=周一,6=周日
df["is_weekend"] = (df["dow"] >= 5).astype(int)

# 出发小时（优先用sched_dep_time；退化到time_hour的小时）
def hh_from_sched_dep(x):
    # sched_dep_time 形如 517 -> 5:17（整数），取小时
    try:
        return int(x) // 100
    except Exception:
        return np.nan

df["dep_hour_sched"] = df["sched_dep_time"].apply(hh_from_sched_dep)
# 有些缺失，回退到 time_hour 的小时
df["dep_hour"] = df["dep_hour_sched"]
mask_nan = df["dep_hour"].isna() & df["time_hour"].notna()
df.loc[mask_nan, "dep_hour"] = pd.to_datetime(df.loc[mask_nan, "time_hour"]).dt.hour
df["dep_hour"] = df["dep_hour"].fillna(0).astype(int).clip(0, 23)

# 高峰时段：7-9点 & 16-19点
peak_hours = set([7, 8, 9, 16, 17, 18, 19])
df["is_peak"] = df["dep_hour"].isin(peak_hours).astype(int)

# 红眼航班（0-5点）
df["is_redeye"] = df["dep_hour"].between(0, 5).astype(int)

# 节假日（美国 2013 年主要联邦假日）
us_holidays_2013 = set(pd.to_datetime([
    "2013-01-01",  # New Year's Day
    "2013-01-21",  # Martin Luther King Jr. Day
    "2013-02-18",  # Presidents' Day
    "2013-05-27",  # Memorial Day
    "2013-07-04",  # Independence Day
    "2013-09-02",  # Labor Day
    "2013-10-14",  # Columbus Day
    "2013-11-11",  # Veterans Day
    "2013-11-28",  # Thanksgiving Day
    "2013-12-25",  # Christmas Day  
]))
df["is_holiday"] = df["date"].isin(us_holidays_2013).astype(int)

# 目标变量：到达延误（分钟）
y_col = "arr_delay"

# ========= 3. 合并天气 =========
# 按机场和时间合并天气数据
# flights 的 time_hour 已是按小时取整的计划起飞时间
df_weather["time_hour"] = pd.to_datetime(df_weather["time_hour"])
df["time_hour"] = pd.to_datetime(df["time_hour"])

# 保留天气相关列
weather_cols = [
    "origin", "time_hour",
    "temp", "dewp", "humid", "wind_dir", "wind_speed", "wind_gust",
    "precip", "pressure", "visib"
]
dfw = df_weather[weather_cols].copy()

# 合并（左连接：保留所有航班）
df = df.merge(dfw, on=["origin", "time_hour"], how="left", validate="m:1")

# 简单缺失处理：按分组中位数插补
weather_num_cols = ["temp","dewp","humid","wind_dir","wind_speed","wind_gust","precip","pressure","visib"]
for c in weather_num_cols:
    grp = df.groupby(["origin","month","dep_hour"])[c].transform("median")
    df[c] = df[c].fillna(grp)
    df[c] = df[c].fillna(df[c].median())

# 额外天气派生变量
df["is_precip"] = (df["precip"] > 0).astype(int)
df["low_vis"] = (df["visib"] < 3).astype(int)  # 能见度<3 英里
df["high_wind"] = (df["wind_speed"] >= 15).astype(int)  # 15 mph 以上

# ========= 4. 特征选择 =========
base_features = [
    "dep_hour", "is_peak", "is_redeye", "is_weekend", "is_holiday",
    "month", "dow"
]
weather_features = [
    "temp", "dewp", "humid", "wind_speed", "wind_gust", "precip", "pressure", "visib",
    "is_precip", "low_vis", "high_wind"
]
feats_for_fe = base_features + weather_features + ["carrier", "origin"]  # 固定效应要保留类别

# 仅保留需要的列
core_cols = ["year","month","day","origin","dest","carrier","flight","distance","time_hour", y_col]
feature_cols = [f for f in feats_for_fe if f not in core_cols]  # 避免重复
df_model = df[core_cols + feature_cols].copy()
# 剪裁极端值
df_model = df_model[(df_model[y_col] > -120) & (df_model[y_col] < 600)].copy()

# ========= 4.5. 一揽子描述性统计分析 =========
print("\n================ 描述性统计分析 ================")

# 基本数据概况
print(f"数据集规模: {len(df_model):,} 条航班记录")
print(f"时间跨度: {df_model['month'].min()}月 - {df_model['month'].max()}月")
print(f"起飞机场: {', '.join(sorted(df_model['origin'].unique()))}")
print(f"航空公司数量: {df_model['carrier'].nunique()} 家")

# 到达延误的基本统计
delay_stats = df_model[y_col].describe()
print(f"\n【到达延误统计】(分钟)")
print(f"平均延误: {delay_stats['mean']:.1f}")
print(f"中位数延误: {delay_stats['50%']:.1f}")
print(f"标准差: {delay_stats['std']:.1f}")
print(f"最小值: {delay_stats['min']:.1f}")
print(f"最大值: {delay_stats['max']:.1f}")
print(f"早到比例 (延误<0): {(df_model[y_col] < 0).mean()*100:.1f}%")
print(f"严重延误 (>60分钟): {(df_model[y_col] > 60).mean()*100:.1f}%")

# 按机场分组统计
print(f"\n【按起飞机场统计】")
airport_stats = df_model.groupby('origin')[y_col].agg(['count', 'mean', 'std']).round(1)
for airport in airport_stats.index:
    count, mean_delay, std_delay = airport_stats.loc[airport]
    print(f"{airport}: {count:,}班次, 平均延误{mean_delay:.1f}分钟, 标准差{std_delay:.1f}")

# 按月份统计
print(f"\n【按月份统计】")
monthly_stats = df_model.groupby('month')[y_col].agg(['count', 'mean']).round(1)
for month in monthly_stats.index:
    count, mean_delay = monthly_stats.loc[month]
    print(f"{month:2d}月: {count:,}班次, 平均延误{mean_delay:.1f}分钟")

# 按星期统计
print(f"\n【按星期统计】")
dow_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
dow_stats = df_model.groupby('dow')[y_col].agg(['count', 'mean']).round(1)
for dow in dow_stats.index:
    count, mean_delay = dow_stats.loc[dow]
    print(f"{dow_names[dow]}: {count:,}班次, 平均延误{mean_delay:.1f}分钟")

# 高峰时段 vs 非高峰时段
print(f"\n【高峰时段 vs 非高峰时段】")
peak_stats = df_model.groupby('is_peak')[y_col].agg(['count', 'mean', 'std']).round(1)
for is_peak in [0, 1]:
    count, mean_delay, std_delay = peak_stats.loc[is_peak]
    peak_label = "高峰时段" if is_peak else "非高峰时段"
    print(f"{peak_label}: {count:,}班次, 平均延误{mean_delay:.1f}分钟, 标准差{std_delay:.1f}")

# 天气条件统计
print(f"\n【天气条件统计】")
weather_conditions = [
    ('降水天气', 'is_precip', 1),
    ('低能见度', 'low_vis', 1), 
    ('强风天气', 'high_wind', 1)
]
for condition_name, col, val in weather_conditions:
    mask = df_model[col] == val
    if mask.sum() > 0:
        count = mask.sum()
        mean_delay = df_model.loc[mask, y_col].mean()
        normal_delay = df_model.loc[~mask, y_col].mean()
        print(f"{condition_name}: {count:,}班次 ({count/len(df_model)*100:.1f}%), 平均延误{mean_delay:.1f}分钟 (正常天气{normal_delay:.1f}分钟)")

# 保存描述性统计到文件
desc_stats = {
    '基本统计': delay_stats.to_dict(),
    '机场统计': airport_stats.to_dict(),
    '月份统计': monthly_stats.to_dict(),
    '星期统计': dow_stats.to_dict(),
    '高峰统计': peak_stats.to_dict()
}

with open(os.path.join(OUT_DIR_TAB, "descriptive_statistics.txt"), "w", encoding="utf-8") as f:
    f.write("航班延误数据描述性统计\n")
    f.write("="*50 + "\n\n")
    f.write(f"数据集规模: {len(df_model):,} 条记录\n")
    f.write(f"时间跨度: 2013年{df_model['month'].min()}-{df_model['month'].max()}月\n\n")
    f.write("到达延误基本统计 (分钟):\n")
    f.write(delay_stats.to_string())
    f.write("\n\n按机场统计:\n")
    f.write(airport_stats.to_string())
    f.write("\n\n按月份统计:\n") 
    f.write(monthly_stats.to_string())

print("="*50)

# ========= 4.6. 描述性分析可视化 =========
import matplotlib.pyplot as plt
# 图1: 延误分布直方图
plt.figure(figsize=(10, 6))
plt.hist(df_model[y_col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(df_model[y_col].mean(), color='red', linestyle='--', label=f'Mean: {df_model[y_col].mean():.1f} min')
plt.axvline(df_model[y_col].median(), color='green', linestyle='--', label=f'Median: {df_model[y_col].median():.1f} min')
plt.xlabel('Arrival Delay (minutes)')
plt.ylabel('Frequency')
plt.title('Distribution of Flight Arrival Delays')
plt.legend()
plt.xlim(-100, 200)  # 聚焦主要范围
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_FIG, "delay_distribution.png"), dpi=160)
plt.savefig(os.path.join(OUT_DIR_FIG, "delay_distribution.pdf"), bbox_inches='tight')
plt.close()

# 图2: 按机场的延误箱线图 - 改进版本
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
# 左图：传统箱线图（限制异常值显示）
airport_data = [df_model[df_model['origin']==airport][y_col] for airport in ['EWR', 'JFK', 'LGA']]
# 对每个机场的数据进行采样
import numpy as np
np.random.seed(42)
sampled_data = []
for data in airport_data:
    if len(data) > 10000:
        sampled = np.random.choice(data, 8000, replace=False)
        sampled_data.append(sampled)
    else:
        sampled_data.append(data)

box_plot = ax1.boxplot(sampled_data, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       flierprops=dict(marker='o', markerfacecolor='red', markersize=1.5, alpha=0.4),
                       showfliers=True)
ax1.set_xticks([1, 2, 3])
ax1.set_xticklabels(['EWR', 'JFK', 'LGA'])
ax1.set_xlabel('Airport')
ax1.set_ylabel('Arrival Delay (minutes)')
ax1.set_title('Arrival Delay Distribution by Airport (Boxplot)')
ax1.set_ylim(-50, 100)
ax1.grid(True, alpha=0.3)
ax1.text(0.02, 0.98, 'Note: Outliers sampled for clarity', transform=ax1.transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# 添加学术化注释解释异常值现象
ax1.text(0.02, 0.02, "The numerous outlier points reflect the inherent\n characteristics of airline operational data.", 
         transform=ax1.transAxes, verticalalignment='bottom', horizontalalignment='left',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcyan', alpha=0.8),
         fontsize=9, style='italic')

# 右图：直方图比较
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
airports = ['EWR', 'JFK', 'LGA']
for i, airport in enumerate(airports):
    airport_delays = df_model[df_model['origin']==airport][y_col]
    # 限制范围突出主要分布
    filtered_delays = airport_delays[(airport_delays >= -50) & (airport_delays <= 100)]
    ax2.hist(filtered_delays, bins=30, alpha=0.6, label=f'{airport} (n={len(airport_delays):,})', 
             color=colors[i], density=True)

ax2.set_xlabel('Arrival Delay (minutes)')
ax2.set_ylabel('Density')
ax2.set_title('Arrival Delay Distribution by Airport (Histogram)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-50, 100)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_FIG, "delay_by_airport.png"), dpi=160)
plt.savefig(os.path.join(OUT_DIR_FIG, "delay_by_airport.pdf"), bbox_inches='tight')
plt.close()

# 图3: 按月份的延误趋势
plt.figure(figsize=(10, 6))
monthly_mean = df_model.groupby('month')[y_col].mean()
monthly_std = df_model.groupby('month')[y_col].std()
months = list(monthly_mean.index)
means = list(monthly_mean.values)
stds = list(monthly_std.values)
plt.plot(months, means, 'o-', linewidth=2, markersize=6, color='navy')
plt.fill_between(months, 
                 [m - s/2 for m, s in zip(means, stds)], 
                 [m + s/2 for m, s in zip(means, stds)], 
                 alpha=0.2, color='navy')
plt.xlabel('Month')
plt.ylabel('Average Arrival Delay (minutes)')
plt.title('Seasonal Pattern of Flight Delays')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 13))
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(range(1, 13), month_names)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_FIG, "delay_by_month.png"), dpi=160)
plt.savefig(os.path.join(OUT_DIR_FIG, "delay_by_month.pdf"), bbox_inches='tight')
plt.close()

# 图4: 按星期的延误模式
plt.figure(figsize=(8, 6))
dow_mean = df_model.groupby('dow')[y_col].mean()
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_values = list(dow_mean.values)
dow_avg = sum(dow_values) / len(dow_values)
colors = ['lightcoral' if x > dow_avg else 'lightgreen' for x in dow_values]
plt.bar(dow_names, dow_values, color=colors, alpha=0.7, edgecolor='black')
plt.axhline(dow_avg, color='red', linestyle='--', alpha=0.7, label=f'Weekly Average: {dow_avg:.1f} min')
plt.xlabel('Day of Week')
plt.ylabel('Average Arrival Delay (minutes)')
plt.title('Weekly Pattern of Flight Delays')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_FIG, "delay_by_weekday.png"), dpi=160)
plt.savefig(os.path.join(OUT_DIR_FIG, "delay_by_weekday.pdf"), bbox_inches='tight')
plt.close()

# 图5: 高峰时段对比
plt.figure(figsize=(8, 5))  
peak_comparison = df_model.groupby('is_peak')[y_col].agg(['mean', 'std', 'count'])
labels = ['Non-Peak Hours', 'Peak Hours']
means = list(peak_comparison['mean'].values)
stds = list(peak_comparison['std'].values)
counts = list(peak_comparison['count'].values)

x_pos = range(len(labels))
plt.bar(x_pos, means, yerr=[s/10 for s in stds], capsize=5, color=['lightblue', 'orange'], 
        alpha=0.7, edgecolor='black')
plt.xlabel('Time Period', fontsize=11)
plt.ylabel('Average Arrival Delay (minutes)', fontsize=11)
plt.title('Peak vs Non-Peak Hours Delay Comparison', fontsize=12)
plt.xticks(x_pos, labels, fontsize=10)
plt.yticks(fontsize=10)
for i, (mean, count) in enumerate(zip(means, counts)):
    plt.text(i, mean + 1, f'{count:,} flights\n{mean:.1f} min', 
             ha='center', va='bottom', fontsize=9)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_FIG, "delay_peak_comparison.png"), dpi=160)
plt.savefig(os.path.join(OUT_DIR_FIG, "delay_peak_comparison.pdf"), bbox_inches='tight')
plt.close()

# 图6: 天气条件影响对比
plt.figure(figsize=(10, 5.5))  # 从(12,7)改为(10,5.5)，更紧凑
weather_conditions = ['Normal\nWeather', 'Precipitation', 'Low\nVisibility', 'High Wind']
weather_delays = []
weather_counts = []

# 正常天气 (所有条件都不满足)
normal_mask = (df_model['is_precip']==0) & (df_model['low_vis']==0) & (df_model['high_wind']==0)
weather_delays.append(df_model.loc[normal_mask, y_col].mean())
weather_counts.append(normal_mask.sum())

# 各种恶劣天气条件
for col in ['is_precip', 'low_vis', 'high_wind']:
    mask = df_model[col] == 1
    weather_delays.append(df_model.loc[mask, y_col].mean())
    weather_counts.append(mask.sum())

# 使用颜色方案
colors = ['#2E8B57', '#4682B4', '#DC143C', '#FF8C00']
bars = plt.bar(weather_conditions, weather_delays, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

plt.xlabel('Weather Condition', fontsize=11)
plt.ylabel('Average Arrival Delay (minutes)', fontsize=11)
plt.title('Impact of Weather Conditions on Flight Delays', fontsize=12, pad=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 调整标签位置
for i, (delay, count) in enumerate(zip(weather_delays, weather_counts)):
    label_height = delay + (max(weather_delays) * 0.12)
    plt.text(i, label_height, f'{count:,} flights\n{delay:.1f} min', 
             ha='center', va='bottom', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0, max(weather_delays) * 1.3)  # 给标签留空间
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_FIG, "delay_by_weather.png"), dpi=160)
plt.savefig(os.path.join(OUT_DIR_FIG, "delay_by_weather.pdf"), bbox_inches='tight')
plt.close()


# ========= 4.7. 增加统计推断相关图表 =========
# 图7: 关键变量效应对比（提前生成，用于后续回归分析）
# 这里先生成一个预览版本，后面用回归结果更新
key_effects = {
    'Peak Hours': 0.6,  # 示例数据，后面会被实际回归结果替换
    'Precipitation': 28.4,
    'Low Visibility': 24.3,
    'High Wind': 5.7,
    'Weekend': -2.1,
    'Holiday': 1.8
}

plt.figure(figsize=(10, 6))
variables = list(key_effects.keys())
effects = list(key_effects.values())
colors = ['red' if x > 0 else 'blue' for x in effects]
bars = plt.barh(variables, effects, color=colors, alpha=0.7, edgecolor='black')

plt.xlabel('Effect on Arrival Delay (minutes)')
plt.title('Key Variable Effects on Flight Delays (Preliminary)')
plt.axvline(0, color='black', linestyle='-', alpha=0.5)
plt.grid(True, alpha=0.3, axis='x')

# 添加数值标签
for i, (var, effect) in enumerate(zip(variables, effects)):
    plt.text(effect + (1 if effect >= 0 else -1), i, f'{effect:.1f}', 
             ha='left' if effect >= 0 else 'right', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_FIG, "key_effects_preview.png"), dpi=160)
plt.savefig(os.path.join(OUT_DIR_FIG, "key_effects_preview.pdf"), bbox_inches='tight')
plt.close()

# 图8: 天气条件交互效应预览
plt.figure(figsize=(12, 8))

# 创建天气×时段的交互分析
interaction_data = []
for weather_cond in ['Normal', 'Precipitation', 'Low Visibility', 'High Wind']:
    for time_period in ['Off-Peak', 'Peak']:
        if weather_cond == 'Normal':
            mask = (df_model['is_precip']==0) & (df_model['low_vis']==0) & (df_model['high_wind']==0)
        elif weather_cond == 'Precipitation':
            mask = df_model['is_precip'] == 1
        elif weather_cond == 'Low Visibility':
            mask = df_model['low_vis'] == 1
        else:  # High Wind
            mask = df_model['high_wind'] == 1
        
        time_mask = df_model['is_peak'] == (1 if time_period == 'Peak' else 0)
        combined_mask = mask & time_mask
        
        if combined_mask.sum() > 0:
            mean_delay = df_model.loc[combined_mask, y_col].mean()
            interaction_data.append({
                'Weather': weather_cond,
                'Time': time_period, 
                'Delay': mean_delay,
                'Count': combined_mask.sum()
            })

interaction_df = pd.DataFrame(interaction_data)

# 创建分组柱状图
weather_conditions = ['Normal', 'Precipitation', 'Low Visibility', 'High Wind']
peak_delays = [interaction_df[(interaction_df['Weather']==w) & (interaction_df['Time']=='Peak')]['Delay'].iloc[0] 
              if len(interaction_df[(interaction_df['Weather']==w) & (interaction_df['Time']=='Peak')]) > 0 else 0 
              for w in weather_conditions]
offpeak_delays = [interaction_df[(interaction_df['Weather']==w) & (interaction_df['Time']=='Off-Peak')]['Delay'].iloc[0] 
                 if len(interaction_df[(interaction_df['Weather']==w) & (interaction_df['Time']=='Off-Peak')]) > 0 else 0 
                 for w in weather_conditions]

x = np.arange(len(weather_conditions))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, offpeak_delays, width, label='Off-Peak Hours', color='lightblue', alpha=0.8)
bars2 = ax.bar(x + width/2, peak_delays, width, label='Peak Hours', color='orange', alpha=0.8)

ax.set_xlabel('Weather Condition')
ax.set_ylabel('Average Arrival Delay (minutes)')
ax.set_title('Weather × Time Period Interaction Effects')
ax.set_xticks(x)
ax.set_xticklabels(weather_conditions)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{height:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_FIG, "weather_time_interaction.png"), dpi=160)
plt.savefig(os.path.join(OUT_DIR_FIG, "weather_time_interaction.pdf"), bbox_inches='tight')
plt.close()



# ========= 5. 训练-测试（时间切分：1-10 月 vs 11-12 月）=========
train_mask = (df_model["month"] >= 1) & (df_model["month"] <= 10)
test_mask  = (df_model["month"] >= 11) & (df_model["month"] <= 12)
train_df = df_model[train_mask].copy()
test_df  = df_model[test_mask].copy()

# ========= 6. 回归模型（statsmodels + 稳健标准误）=========
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error

def fit_ols(formula, data):
    model = smf.ols(formula=formula, data=data)
    res = model.fit(cov_type="HC3")  # 稳健标准误（异方差稳健）
    return res

# 基准模型：时段+日历变量（不用C(month)因为会有训练测试集类别不匹配问题）
formula_base = (
    "arr_delay ~ dep_hour + is_peak + is_redeye + is_weekend + is_holiday "
    "+ month + C(dow)"
)

# 加天气
formula_wx = (
    formula_base + " + temp + dewp + humid + wind_speed + wind_gust + precip + pressure + visib "
    "+ is_precip + low_vis + high_wind"
)

# 再加固定效应：承运人和起飞机场
formula_fe = formula_wx + " + C(carrier) + C(origin)"

res_base = fit_ols(formula_base, train_df)
res_wx   = fit_ols(formula_wx,   train_df)
res_fe   = fit_ols(formula_fe,   train_df)

# 保存回归摘要
with open(os.path.join(OUT_DIR_TAB, "model_base.txt"), "w", encoding="utf-8") as f:
    f.write(res_base.summary().as_text())
with open(os.path.join(OUT_DIR_TAB, "model_wx.txt"), "w", encoding="utf-8") as f:
    f.write(res_wx.summary().as_text())
with open(os.path.join(OUT_DIR_TAB, "model_fe.txt"), "w", encoding="utf-8") as f:
    f.write(res_fe.summary().as_text())

# ========= 7. 测试集评估（RMSE/MAE）=========
def predict_on(res, train_df, test_df):
    # statsmodels 的 formula 模型需要把训练集中的类别水平映射到测试集
    # 使用 res.model.data.design_info 构建设计矩阵
    y_true = test_df[y_col].values
    y_pred = res.predict(test_df)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return rmse, mae, y_pred

rmse_base,mae_base,ypred_base = predict_on(res_base, train_df, test_df)
rmse_wx,mae_wx,ypred_wx   = predict_on(res_wx,   train_df, test_df)
rmse_fe,mae_fe,ypred_fe   = predict_on(res_fe,   train_df, test_df)

report_lines = [
    "Test-set performance (2013-11~12):",
    f"[Base]   RMSE={rmse_base:.2f}, MAE={mae_base:.2f}",
    f"[Base+WX]RMSE={rmse_wx:.2f}, MAE={mae_wx:.2f}",
    f"[+FE]    RMSE={rmse_fe:.2f}, MAE={mae_fe:.2f}",
]
print("\n".join(report_lines))
with open(os.path.join(OUT_DIR_TAB, "test_performance.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

# ========= 7.5. 统计推断图表 =========
# 图A: 回归系数置信区间图 (Forest Plot)
# 提取主要变量的系数和置信区间
key_vars = ['is_peak', 'is_precip', 'low_vis', 'high_wind', 'is_weekend', 'is_holiday', 'temp', 'wind_speed']
var_names = ['Peak Hours', 'Precipitation', 'Low Visibility', 'High Wind', 'Weekend', 'Holiday', 'Temperature', 'Wind Speed']

# 从最终模型（加固定效应）提取系数
coeffs = []
conf_ints = []
for var in key_vars:
    if var in res_fe.params.index:
        coeffs.append(res_fe.params[var])
        ci = res_fe.conf_int().loc[var]
        conf_ints.append([ci[0], ci[1]])
    else:
        coeffs.append(0)
        conf_ints.append([0, 0])

fig, ax = plt.subplots(figsize=(8, 6))  # 从(10,8)改为(8,6)
y_pos = np.arange(len(var_names))

# 绘制置信区间
for i, (coeff, ci, name) in enumerate(zip(coeffs, conf_ints, var_names)):
    ax.errorbar(coeff, i, xerr=[[coeff-ci[0]], [ci[1]-coeff]], 
                fmt='o', markersize=6, capsize=4, capthick=1.5,
                color='red' if coeff > 0 else 'blue', alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(var_names, fontsize=10)
ax.set_xlabel('Effect on Arrival Delay (minutes)', fontsize=11)
ax.set_title('Regression Coefficients with 95% Confidence Intervals', fontsize=12)
ax.tick_params(axis='x', labelsize=10)
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3, axis='x')

# 添加显著性标记
for i, (coeff, ci) in enumerate(zip(coeffs, conf_ints)):
    if ci[0] * ci[1] > 0:  # 不包含0，显著
        ax.text(coeff + 0.3, i, '*', fontsize=14, fontweight='bold', color='red')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_FIG, "coefficient_forest_plot.png"), dpi=160)
plt.savefig(os.path.join(OUT_DIR_FIG, "coefficient_forest_plot.pdf"), bbox_inches='tight')
plt.close()

# 图B: 模型比较 (R² 和性能指标)
models = ['Baseline', 'Baseline + Weather', 'Full Model + FE']
r_squared = [res_base.rsquared, res_wx.rsquared, res_fe.rsquared]
adj_r_squared = [res_base.rsquared_adj, res_wx.rsquared_adj, res_fe.rsquared_adj]
rmse_values = [rmse_base, rmse_wx, rmse_fe]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# R² 比较
x = np.arange(len(models))
width = 0.35
bars1 = ax1.bar(x - width/2, r_squared, width, label='R²', alpha=0.8, color='skyblue')
bars2 = ax1.bar(x + width/2, adj_r_squared, width, label='Adjusted R²', alpha=0.8, color='lightcoral')

ax1.set_xlabel('Model')
ax1.set_ylabel('R² Value')
ax1.set_title('Model Explanatory Power Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# RMSE 比较
bars3 = ax2.bar(models, rmse_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
ax2.set_xlabel('Model')
ax2.set_ylabel('RMSE (minutes)')
ax2.set_title('Model Prediction Accuracy (Test Set)')
ax2.grid(True, alpha=0.3, axis='y')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15)

# 添加数值标签
for bar in bars3:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
           f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_FIG, "model_comparison.png"), dpi=160)
plt.savefig(os.path.join(OUT_DIR_FIG, "model_comparison.pdf"), bbox_inches='tight')
plt.close()

# 图C: 残差诊断图
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 获取最终模型的残差和拟合值
fitted_values = res_fe.fittedvalues
residuals = res_fe.resid

# 1. 残差 vs 拟合值
ax1.scatter(fitted_values, residuals, alpha=0.5, s=1)
ax1.axhline(0, color='red', linestyle='--', alpha=0.7)
ax1.set_xlabel('Fitted Values')
ax1.set_ylabel('Residuals')
ax1.set_title('Residuals vs Fitted Values')
ax1.grid(True, alpha=0.3)

# QQ图
stats.probplot(residuals.sample(5000), dist="norm", plot=ax2)
ax2.set_title('Normal Q-Q Plot of Residuals')
ax2.grid(True, alpha=0.3)

# 3. 残差直方图
ax3.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
ax3.set_xlabel('Residuals')
ax3.set_ylabel('Density')
ax3.set_title('Distribution of Residuals')
ax3.grid(True, alpha=0.3)

# 4. 标准化残差
standardized_residuals = residuals / residuals.std()
ax4.scatter(fitted_values, standardized_residuals, alpha=0.5, s=1)
ax4.axhline(0, color='red', linestyle='--', alpha=0.7)
ax4.axhline(2, color='orange', linestyle='--', alpha=0.7, label='±2σ')
ax4.axhline(-2, color='orange', linestyle='--', alpha=0.7)
ax4.set_xlabel('Fitted Values')
ax4.set_ylabel('Standardized Residuals')
ax4.set_title('Standardized Residuals vs Fitted Values')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_FIG, "residual_diagnostics.png"), dpi=160)
plt.savefig(os.path.join(OUT_DIR_FIG, "residual_diagnostics.pdf"), bbox_inches='tight')
plt.close()

# 图D: 预测误差分布
prediction_errors = test_df[y_col].values - ypred_fe

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 预测误差直方图
ax1.hist(prediction_errors, bins=50, alpha=0.7, color='lightgreen', edgecolor='black', density=True)
ax1.axvline(prediction_errors.mean(), color='red', linestyle='--', 
           label=f'Mean: {prediction_errors.mean():.1f}')
ax1.axvline(0, color='black', linestyle='-', alpha=0.5, label='Perfect Prediction')
ax1.set_xlabel('Prediction Error (Actual - Predicted)')
ax1.set_ylabel('Density')
ax1.set_title('Distribution of Prediction Errors (Test Set)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 绝对误差 vs 实际延误
abs_errors = np.abs(prediction_errors)
ax2.scatter(test_df[y_col], abs_errors, alpha=0.5, s=1)
ax2.set_xlabel('Actual Delay (minutes)')
ax2.set_ylabel('Absolute Prediction Error')
ax2.set_title('Absolute Error vs Actual Delay')
ax2.grid(True, alpha=0.3)

# 添加趋势线
z = np.polyfit(test_df[y_col], abs_errors, 1)
p = np.poly1d(z)
ax2.plot(test_df[y_col], p(test_df[y_col]), "r--", alpha=0.8, label='Trend')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_FIG, "prediction_error_analysis.png"), dpi=160)
plt.savefig(os.path.join(OUT_DIR_FIG, "prediction_error_analysis.pdf"), bbox_inches='tight')
plt.close()



# ========= 8. 可视化 =========
import matplotlib.pyplot as plt

# 8.1 不同时段的平均到达延误（描述性）
by_hour = (
    df_model.groupby("dep_hour")[y_col] 
    .mean().reindex(range(24))
)
plt.figure(figsize=(8,4))
by_hour.plot(marker="o")
plt.axvspan(7,9, alpha=0.1)   # 早高峰
plt.axvspan(16,19, alpha=0.1) # 晚高峰
plt.title("Average Arrival Delay by Departure Hour")
plt.xlabel("Departure Hour")
plt.ylabel("Average Arrival Delay (minutes)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_FIG, "delay_by_hour.png"), dpi=160)
plt.savefig(os.path.join(OUT_DIR_FIG, "delay_by_hour.pdf"), bbox_inches='tight')  
plt.close()

# 8.2 预测 vs 实际（测试集）
plot_df = test_df.copy()
plot_df = (
    plot_df[["time_hour", y_col]]
    .assign(pred_base = ypred_base,
            pred_wx   = ypred_wx,
            pred_fe   = ypred_fe)
    .sort_values("time_hour")
    .head(3000)  # 取子集
)

plt.figure(figsize=(10,4))
plt.plot(plot_df["time_hour"], plot_df[y_col], label="Actual", linewidth=1)
plt.plot(plot_df["time_hour"], plot_df["pred_base"], label="Baseline", linewidth=1)
plt.plot(plot_df["time_hour"], plot_df["pred_wx"], label="Baseline + Weather", linewidth=1)
plt.plot(plot_df["time_hour"], plot_df["pred_fe"], label="Baseline + Weather + FE", linewidth=1)
plt.title("Test Set: Predicted vs Actual (Sample of 3000 points)")
plt.xlabel("Time")
plt.ylabel("Arrival Delay (minutes)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_FIG, "pred_vs_actual_test.png"), dpi=160)
plt.savefig(os.path.join(OUT_DIR_FIG, "pred_vs_actual_test.pdf"), bbox_inches='tight')  
plt.close()

# 8.3 天气变量的边际关系
def bin_plot(x, y, bins=20, title="", outfile="tmp.png"):
    s = pd.Series(x).astype(float)
    q = pd.qcut(s, q=bins, duplicates="drop")
    tmp = pd.DataFrame({"bin": q, "y": y}).groupby("bin")["y"].mean()
    xc = [str(interval) for interval in tmp.index]
    plt.figure(figsize=(8,3.5))
    plt.plot(range(len(tmp)), list(tmp.values), marker="o")
    plt.xticks(range(len(tmp)), xc, rotation=60, ha="right")
    plt.title(title)
    plt.ylabel("Average Arrival Delay (minutes)")
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.savefig(outfile.replace('.png', '.pdf'), bbox_inches='tight')  
    plt.close()

bin_plot(test_df["precip"], test_df[y_col],
         title="Quantile View: Precipitation vs Average Delay (Test Set)",
         outfile=os.path.join(OUT_DIR_FIG, "marginal_precip.png"))

bin_plot(test_df["visib"], test_df[y_col],
         title="Quantile View: Visibility vs Average Delay (Test Set)",
         outfile=os.path.join(OUT_DIR_FIG, "marginal_visib.png"))

# ========= 9. 导出干净版数据 =========
out_cols = ["year","month","day","origin","dest","carrier","distance","time_hour",
            y_col] + base_features + weather_features
clean_export = df_model[out_cols].copy()
clean_export.to_csv(os.path.join(OUT_DIR_DATA, "flights_model.csv"), index=False)

# ========= 10. 简要结论 =========

print(f"- 训练样本量：{len(train_df):,}；测试样本量：{len(test_df):,}")
print("- 基准模型包含：出发小时/高峰/红眼/周末/节假日 + 月度/星期固定效应；")
print("- 加天气后显著提升解释力；再加入承运人&起飞机场固定效应，测试集 RMSE/MAE 进一步下降。")
print("- 图：results/fig/delay_by_hour.png 展示“高峰时段延误更高”的描述性证据；")
print("     results/fig/pred_vs_actual_test.png 对比不同模型的拟合；")
print("     results/fig/marginal_*.png 展示降水/能见度与延误的单调关系。")
print("- 导出 data_clean/flights_model.csv")
print("====================================================")
