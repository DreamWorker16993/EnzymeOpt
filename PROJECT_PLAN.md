# EnzymeOpt 项目计划

## 1. 项目目标与第一版边界

EnzymeOpt 使用主动学习和最优实验设计，在尽可能少的底物浓度测量下估计 Michaelis–Menten 模型的 `KM` 和 `Vmax`。第一版将提供带加性 Gaussian noise 的虚拟数据、非线性最小二乘拟合、random、log-spaced 和序贯局部 D-optimal 三种浓度选择策略，以及可复现的 Monte Carlo 对比实验。

第一版不包含网页、图形界面、神经网络、真实实验设备接入、Bayesian optimal design 或复杂噪声模型。

## 2. 已确认的基线约定

- 模型：`v(S) = Vmax * S / (KM + S) + epsilon`。
- 噪声：独立同分布的加性 Gaussian noise，基线 `sigma = 0.05 * Vmax`；模拟产生的负速率不裁剪。
- 基线真值：`KM = 1.0`、`Vmax = 1.0`。
- 浓度范围：基线为 `[0.05 * KM, 20 * KM]`。
- 候选集合：在浓度上下限之间生成 200 个 log-spaced 候选点。
- 重复浓度：允许。
- 初始探索设计：浓度下限、几何中点和浓度上限，共 3 次测量。
- 测量预算：`3, 4, 6, 8, 12, 16, 24`。
- random：在 log-concentration 空间均匀抽样。
- log-spaced：针对每个测量预算独立生成完整设计，不要求预算之间嵌套。
- D-optimal：采用基于当前参数估计的 sequential locally D-optimal design。
- 拟合：在 log-parameter 空间执行 nonlinear least squares，以保证 `KM > 0`、`Vmax > 0`。
- 置信区间：第一版使用 Jacobian 局部线性近似的 95% CI，并报告 empirical coverage。
- Monte Carlo：开发运行使用 20–50 个 replicates，正式基准使用至少 1,000 个 replicates。
- 可复现性：所有随机过程均从显式主 seed 派生，不依赖隐式全局随机状态。

## 3. Milestones

### Milestone 1：项目骨架与开发工具

**目标**

建立可安装的 `src` 布局、pytest 测试入口、基础包元数据和最小使用说明，为后续模块提供稳定边界。

**需要修改的文件**

- `pyproject.toml`
- `README.md`
- `src/enzymeopt/__init__.py`
- `tests/test_package.py`
- `.gitignore`

**输入和输出**

- 输入：本项目计划、支持的 Python 版本和核心依赖选择。
- 输出：可安装的 `enzymeopt` 包、可运行的 pytest 测试套件、清晰的开发安装说明。

**完成条件**

- 包可以从 `src` 布局正确导入。
- pytest 能发现并运行测试。
- 依赖被明确区分为运行依赖和开发依赖。
- 输出目录、缓存和虚拟环境不会被误提交。

**需要运行的测试**

- `pytest tests/test_package.py`
- `pytest`
- 包安装与导入 smoke test。

### Milestone 2：配置、数据结构与随机数管理

**目标**

定义实验配置、观测数据、拟合结果和实验结果的数据结构，并建立与执行顺序无关的分层 seed 派生机制。

**需要修改的文件**

- `src/enzymeopt/config.py`
- `src/enzymeopt/results.py`
- `src/enzymeopt/randomness.py`
- `tests/test_config.py`
- `tests/test_results.py`
- `tests/test_reproducibility.py`

**输入和输出**

- 输入：真值、噪声水平、浓度范围、候选点数、预算、replicate 数量、策略名和主 seed。
- 输出：通过验证的不可歧义配置、统一结果对象、可重复生成的 replicate/strategy/noise 随机数流。

**完成条件**

- 非法范围、非正参数、无效预算和未知策略会产生明确错误。
- 配置可序列化并无损恢复。
- 相同主 seed 和标签产生相同随机流。
- 调换策略执行顺序不会改变任一策略的随机流。

**需要运行的测试**

- `pytest tests/test_config.py`
- `pytest tests/test_results.py`
- `pytest tests/test_reproducibility.py`
- `pytest`

### Milestone 3：Michaelis–Menten 模型与灵敏度

**目标**

实现 Michaelis–Menten 均值函数、原参数与 log 参数转换，以及供拟合和最优设计使用的解析 Jacobian。

**需要修改的文件**

- `src/enzymeopt/model.py`
- `tests/test_model.py`

**输入和输出**

- 输入：底物浓度数组、`KM` 和 `Vmax`，或对应的 log 参数。
- 输出：期望反应速率、参数转换结果、对 `KM` 和 `Vmax` 的灵敏度矩阵。

**完成条件**

- `S = 0` 时速率为零，高浓度时速率趋近 `Vmax`。
- 负浓度及非正物理参数被拒绝。
- 标量和数组输入行为一致且有文档说明。
- 解析 Jacobian 与有限差分结果在规定容差内一致。

**需要运行的测试**

- `pytest tests/test_model.py`
- `pytest`

### Milestone 4：虚拟数据生成

**目标**

生成无噪声或带同方差加性 Gaussian noise 的虚拟酶动力学观测，并完整保留生成元数据。

**需要修改的文件**

- `src/enzymeopt/simulation.py`
- `tests/test_simulation.py`
- `tests/test_reproducibility.py`

**输入和输出**

- 输入：浓度、真实 `KM`、真实 `Vmax`、噪声标准差和显式随机数生成器。
- 输出：浓度、无噪声均值、观测速率、噪声设置和真值元数据。

**完成条件**

- 零噪声输出与模型值完全一致。
- 相同 seed 的观测完全一致。
- Gaussian noise 的经验均值和标准差在预设统计容差内。
- 负观测值被保留，不做静默裁剪。

**需要运行的测试**

- `pytest tests/test_simulation.py`
- `pytest tests/test_reproducibility.py`
- `pytest`

### Milestone 5：非线性拟合与置信区间

**目标**

在 log-parameter 空间拟合 `KM` 和 `Vmax`，计算协方差与 95% CI，并对不可识别、未收敛和边界情况返回结构化诊断。

**需要修改的文件**

- `src/enzymeopt/fitting.py`
- `src/enzymeopt/results.py`
- `tests/test_fitting.py`

**输入和输出**

- 输入：浓度、观测速率、确定性初始值规则和拟合选项。
- 输出：`KM`、`Vmax`、残差、Jacobian、协方差、95% CI、收敛状态和失败原因。

**完成条件**

- 无噪声且设计充分的数据能在规定容差内恢复真值。
- 输出参数始终为正。
- `n <= 2`、秩不足或矩阵病态时不会返回误导性的有效 CI。
- 相同输入产生相同拟合结果。
- 拟合失败作为结果状态保留，不被静默丢弃。

**需要运行的测试**

- `pytest tests/test_fitting.py`
- 无噪声参数恢复测试。
- 低噪声参数恢复测试。
- `pytest`

### Milestone 6：Random 与 Log-spaced 设计策略

**目标**

建立统一的浓度选择策略接口，并实现公平可比的 random 与 log-spaced 基线策略。

**需要修改的文件**

- `src/enzymeopt/designs/__init__.py`
- `src/enzymeopt/designs/base.py`
- `src/enzymeopt/designs/random.py`
- `src/enzymeopt/designs/log_spaced.py`
- `tests/test_designs.py`

**输入和输出**

- 输入：浓度范围、测量预算、已有观测、候选集合和可选随机数生成器。
- 输出：位于允许范围内的下一浓度或完整静态浓度设计。

**完成条件**

- random 在 log-concentration 空间均匀采样且可复现。
- log-spaced 包含指定端点并具有正确的对数间距。
- 每个预算可独立生成完整 log-spaced 设计。
- 两种策略遵守相同的范围和重复浓度规则。

**需要运行的测试**

- `pytest tests/test_designs.py -k "random or log_spaced"`
- random 边界与可重复性测试。
- log-spaced 端点与间距测试。
- `pytest`

### Milestone 7：Fisher Information 与 D-optimal 设计

**目标**

实现 Fisher information、稳定的 log-determinant 准则，以及使用当前拟合参数逐点选浓度的 sequential locally D-optimal 策略。

**需要修改的文件**

- `src/enzymeopt/information.py`
- `src/enzymeopt/designs/d_optimal.py`
- `tests/test_information.py`
- `tests/test_designs.py`

**输入和输出**

- 输入：当前参数估计、已有设计、噪声水平和 log-spaced 候选浓度集合。
- 输出：当前信息矩阵、各候选点的 D-optimal 分数、确定性选出的下一浓度。

**完成条件**

- 信息矩阵与手工计算结果一致。
- 数值实现使用稳定的 log-determinant 或等价分解。
- 选中点确实最大化候选集合上的增量准则。
- 分数并列时采用确定性的 tie-breaking。
- 奇异或病态信息矩阵得到明确诊断或规定的数值保护。

**需要运行的测试**

- `pytest tests/test_information.py`
- `pytest tests/test_designs.py -k d_optimal`
- 解析信息矩阵手算测试。
- D-optimal 候选枚举对照测试。
- `pytest`

### Milestone 8：序贯实验执行器

**目标**

将初始探索设计、模拟、拟合和下一浓度选择连接为统一的单次序贯实验流程，并支持所有测量预算。

**需要修改的文件**

- `src/enzymeopt/experiment.py`
- `src/enzymeopt/results.py`
- `tests/test_experiment.py`
- `tests/test_reproducibility.py`

**输入和输出**

- 输入：一个已验证配置、策略实例、真实参数和 replicate 随机数流。
- 输出：每一步的浓度、观测、参数估计、CI、设计诊断和最终实验状态。

**完成条件**

- 三种策略通过同一个实验接口运行。
- 序贯策略从共同的 3 点初始探索设计开始。
- 每个预算对应的测量数量准确。
- 中间拟合失败按照预定义规则记录和处理。
- 相同 seed 的完整轨迹完全一致。

**需要运行的测试**

- `pytest tests/test_experiment.py`
- `pytest tests/test_reproducibility.py`
- 三种策略端到端 smoke test。
- 各测量预算计数测试。
- `pytest`

### Milestone 9：指标与 Monte Carlo 比较框架

**目标**

运行多策略、多预算和多 replicate 的公平比较，计算参数误差、CI 质量、测量数量及拟合成功率。

**需要修改的文件**

- `src/enzymeopt/metrics.py`
- `src/enzymeopt/monte_carlo.py`
- `src/enzymeopt/results.py`
- `tests/test_metrics.py`
- `tests/test_monte_carlo.py`
- `tests/test_reproducibility.py`

**输入和输出**

- 输入：实验配置、策略集合、预算集合、replicate 数量和主 seed。
- 输出：replicate 级明细与聚合结果，包括 `KM`/`Vmax` relative error、CI width、relative CI width、coverage、measurement count 和 fitting success rate。

**完成条件**

- mean、median、分位数和失败率的分母规则明确且正确。
- 失败拟合不会被转换为看似有效的零误差或零宽度。
- 同一 replicate 下各策略使用公平且可追踪的随机条件。
- 改变策略执行顺序不会改变结果。
- 不以“D-optimal 每次都优于 random”作为硬编码测试假设。

**需要运行的测试**

- `pytest tests/test_metrics.py`
- `pytest tests/test_monte_carlo.py`
- `pytest tests/test_reproducibility.py`
- 小规模固定 seed Monte Carlo smoke test。
- 手算聚合结果对照测试。
- `pytest`

### Milestone 10：命令行、结果持久化与正式基准

**目标**

提供从配置文件运行完整实验的命令行入口，保存可审计结果，并完成第一版正式 Monte Carlo 基准和文档。

**需要修改的文件**

- `src/enzymeopt/cli.py`
- `configs/baseline.toml`
- `scripts/run_benchmark.py`
- `tests/test_cli.py`
- `tests/test_results_io.py`
- `tests/test_experiment_smoke.py`
- `README.md`
- `.gitignore`

**输入和输出**

- 输入：TOML 配置文件、可选 seed 覆盖和输出目录。
- 输出：CSV/JSON 格式的 replicate 明细、聚合指标、完整配置、seed、软件版本、运行状态，以及可选的静态结果图。

**完成条件**

- 基线配置可从命令行端到端运行。
- 保存后的结果可无损重新加载。
- 输出包含重现实验所需的全部元数据。
- 开发 smoke run 和至少 1,000 replicates 的正式基准均完成。
- README 准确说明数学假设、安装方式、运行方式和结果解释。
- 全部测试与验证通过，工作区不存在意外生成文件。

**需要运行的测试**

- `pytest tests/test_cli.py`
- `pytest tests/test_results_io.py`
- `pytest tests/test_experiment_smoke.py`
- 固定 seed 的完整端到端复现测试。
- `pytest`
- 使用 `configs/baseline.toml` 运行开发规模 smoke benchmark。
- 使用正式配置运行至少 1,000 replicates，并验证输出完整性。

## 4. 跨 Milestone 交付规则

- 每个 milestone 必须作为独立、可审查的交付单元完成。
- 每次改动必须同步新增或更新相关测试。
- 在创建 milestone 对应的 Git commit 前，必须运行该 milestone 列出的专项测试和完整测试套件。
- 每个 milestone 至少创建一个对应的 Git commit；若拆分多个 commit，每个 commit 都必须保持测试通过。
- 不得通过忽略失败样本来美化策略表现；所有失败状态必须进入结果记录与汇总。
- 所有公共接口和随机行为必须有文档，并能通过固定 seed 重现。
- 后续实现若需要改变本计划中的数学假设或公平比较规则，应先更新本文件并单独提交。
