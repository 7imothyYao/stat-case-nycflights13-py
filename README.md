# 航班延误分析项目

本项目分析航班延误与天气、时段等因素的关系，基于 nycflights13 数据集。


### 1. 克隆到本地
```bash
git clone https://github.com/7imothyYao/stat-case-nycflights13-py.git
cd stat-case-nycflights13-py
```

### 2. 创建并激活虚拟环境
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows (PowerShell/CMD):
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 运行分析
```bash
python flight_delay_project.py
```

## 运行结果

运行完成后，你将得到：

- **数据文件**: `data_clean/flights_model.csv` - 清理后的航班数据
- **图表**: `results/fig/` 目录下的 PNG 和 PDF 格式图表
  - `delay_by_hour.pdf` - Average delay patterns by departure hour
  - `pred_vs_actual_test.pdf` - Model prediction performance comparison
  - `marginal_precip.pdf` - Effect of precipitation on flight delays
  - `marginal_visib.pdf` - Effect of visibility on flight delays
- **统计结果**: `results/tables/` 目录下的回归分析结果
  - `model_base.txt` - 基准模型回归结果
  - `model_wx.txt` - 加入天气变量的模型结果
  - `test_performance.txt` - 模型性能指标

## 项目结构

```text
├── flight_delay_project.py    # 主分析脚本
├── requirements.txt           # 项目依赖
├── data_clean/               # 输出：清理后数据
├── results/
│   ├── fig/                  # 输出：图表
│   └── tables/               # 输出：回归结果
└── README.md                 # 项目说明
```

## 分析内容

1. **数据清洗与特征工程**
   - 处理缺失值和异常值
   - 创建时间特征（高峰时段、红眼航班、节假日等）
   - 天气数据合并和清洗

2. **统计建模**
   - 基准模型：时段和日历效应
   - 扩展模型：加入天气变量
   - 使用稳健标准误(HC3)处理异方差

3. **模型评估**
   - 时间切分验证（1-10月训练，11-12月测试）
   - RMSE 和 MAE 性能指标
   - 预测 vs 实际对比可视化

4. **结果可视化**
   - 按小时延误模式分析
   - 天气因素边际效应图
   - 模型预测效果对比