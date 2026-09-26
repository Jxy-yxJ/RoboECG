# ECG 电极定位与机械臂贴放 Pipeline

状态：v1.0（2026-09-18）
地位：本文件是 ECG 项目的唯一规划入口；后续 `ECG_M*_FINDINGS.md` 均引用本文件的里程碑编号与验收标准。
输入：临床 12 导联位置表与文献调研（见第 9 节参考）。
上游工作：`farus_thyroid_isaac/`（FARUS 甲状腺定位复现，M0–M4 已完成）。

---

## 0. 已确认决策

| 决策项 | 选择 | 说明 |
|---|---|---|
| 机器人平台 | **UR3 先行，Z1 Pro 后续迁移** | 直接复用甲状腺的 UR3 + Lula IK + 安全规划；M5 迁 Z1 |
| 电极范围 | **仅 V1–V6 胸导联（6 点）** | 肢体导联留待后续（可选 Mason-Likar 上躯干方案） |
| 患者体位 | **仰卧**（临床标准） | 新建仰卧姿态与床体资产；胸部朝上，机器人自上而下贴放 |
| 电极与末端 | **干电极按压**（首版） | 弹簧干电极 + FT 传感器；无需粘附/剥离，可反复执行 |
| 感知路线 | **几何外推 + 合成数据检测器（并行对比）** | M3a 规则外推；M3b Isaac 合成数据训练 landmark 检测器 |
| 深度融合 | **M2 列入首版** | 目标生成阶段即用 RGB-D 皮肤点 + 法向修正 |

首版交付（v1）：Isaac Sim 内 `仰卧场景 → 目标生成 → 到达 → 按压 → 自检 → 评估` 全闭环。
后续（v1.5+）：Z1 Pro 迁移、真机力控、ECG 信号侧错位检测、VLA 视觉伺服。

---

## 1. 任务定义

**目标**：机械臂在仰卧患者的胸廓表面，按临床标准定位并贴放 V1–V6 六个干电极，贴放后能自检位置偏差并自动重贴。

**v1 范围**
- 在 Isaac Sim 中完成全部感知、定位、规划、执行、评估；
- 不采集真实 ECG 信号（仅预留接口）；
- 不做肢体导联、不做湿电极剥离/粘附；
- 不做多患者真人实验（M5 之后）。

**非目标**
- 心电诊断算法本身；
- 心电逆问题/体表电位标测（那是"检测已贴电极"文献的应用场景，本项目只借用其定位方法做验证）。

---

## 2. 临床标准与硬约束

### 2.1 十二导联与 V1–V6 标准位置

| 电极 | 纵向（肋间） | 横向（参考线） |
|---|---|---|
| V1 | 第 4 肋间 | 胸骨**右**缘 |
| V2 | 第 4 肋间 | 胸骨**左**缘 |
| V3 | V2–V4 连线中点 | V2–V4 连线中点 |
| V4 | 第 5 肋间 | 左**锁骨中线** |
| V5 | 与 V4 同一水平 | 左**腋前线** |
| V6 | 与 V4 同一水平 | 左**腋中线** |

肢体导联（4 个，v1 不做）：RA/LA/RL/LL 分别位于双手腕、双脚踝；监护场景常用 Mason-Likar 改放躯干（锁骨下 + 髂前上棘附近）。

### 2.2 精度与解剖约束

| 约束 | 数值/事实 | 来源 |
|---|---|---|
| 临床可接受定位误差 | ~1 cm | 临床共识；Shashank 目标 ±0.5 cm |
| 2 cm 位移即改变波形/诊断 | 前胸导联位移对 ECG 形态影响显著 | DOI 10.1007/s11517-013-1115-9 |
| 肋间不可见 | 需靠胸骨角（第 2 肋）、锁骨中线、腋前线/中线 + 人体测量比例外推；男性乳头线 ≈ 第 4 肋间 | 临床解剖 |
| 呼吸位移 | 胸壁 z 向 2.5–9.7 mm（7 人） | El Ghebouli 2025, Table 2 |
| 体型影响 | BMI 19.6–26.3 下电极条走向/间距差异显著 | El Ghebouli 2025 |
| 人工贴放错误 | 护士/技师/医师贴放错误率显著，自动化有明确动机 | DOI 10.1016/s0022-0736(96)80080-x, DOI 10.1177/0017896912472328 |

### 2.3 干电极接触要求（v1 仿真代理）

- 按压深度 2 mm（目标），接触力 0.5–2 N（可配置，硬上限 3 N）；
- 接触法向与皮肤法向夹角 ≤ 25°（沿用甲状腺 M4 的法向约束经验）；
- 电极外径 < V1/V2 间距（~4 cm），避免触碰已贴点。

---

## 3. 文献调研总结

### 3.1 路线对照

| 路线 | 代表工作 | 方法 | 结果 | 本项目取用 |
|---|---|---|---|---|
| 骨骼 + 体型匹配 | Shashank et al. 2018（DOI 10.1007/978-3-319-60483-1_55） | Kinect V2 骨骼 → 数据库匹配相似体型 → 尺度变换 | <1 cm（GT 定义偏弱） | 规则生成 + "体型模板迁移"备选策略 |
| RGB-D 几何规则 | Țichindelean et al. 2018 | RGB-D + 解剖几何 | 泛化差（相机位姿/距离/胖瘦） | 教训：必须体型自适应、相机外参显式标定 |
| 影像预测理想位置 | **Li et al., MedIA 2025**（DOI 10.1016/j.media.2025.103472；arXiv:2408.13945；代码 `github.com/lileitech/12lead_ECG_electrode_localizer`） | 电极作为 torso keypoints 子集 + topology 约束，从心脏 MRI 稀疏躯干轮廓回归 10 电极 | **1.24±0.29 cm，2 s**（投影法 1.48±0.36 cm / 30–35 min） | 设计原则 P2：输出必须约束在躯干表面流形上 |
| 深度相机 + 解剖 landmark | JBHI 2024（DOI 10.1109/JBHI.2024.3520486） | depth → DL 检测 T1–T9/胸骨尖/肺尖 → 胸廓坐标系 → 电极 | landmark cm 级，200 例 | 设计原则 P1 + M3b 的直接模板 |
| 独立电极真值 | Zenodo 10.5281/zenodo.20086105（Bender et al., CC-BY-4.0） | 25 躯干模型 + 标准 12 导联电极位置 → 标定/验证纵向与横向规则 | V1→V4 落差 86.7 ± 16.7 mm；V4–V6 105.6 ± 23.7 mm | **已用于修正 M1 规则**（`docs/ECG_GT_VALIDATION.md`） |
| 检测已贴电极（3D 相机） | El Ghebouli et al., Front. Physiol. 2025（DOI 10.3389/fphys.2025.1504319） | YOLOv8-X + RealSense D415 12 视角 + ICP + 自动编号 | 幻影 <2 mm；真人 2.61±1.2–5.78±3.09 mm | M4 贴后视觉验证 |
| 检测已贴电极（3D 相机） | Bayer et al., Sensors 2023（DOI 10.3390/s23125552） | RealSense SR300，14 视角 ~270°，颜色标记 + ICP | 2.0±1.5 mm | 多视角验证备选 |
| 躯干三维摄影 | DOI 10.1016/j.jelectrocard.2017.08.035 | 3D 摄影重建躯干几何 | — | 表面几何重建参考 |
| 机器人肋间超声 | TMRB 2025（DOI 10.1109/TMRB.2025.3550663）；TASE 2024（DOI 10.1109/TASE.2024.3411784）；IROS 2025（DOI 10.1109/IROS60139.2025.11246299） | 机器人沿肋间自主扫描：肋软骨分割、触觉/力引导 | — | 肋间感知 + 力控参考 |
| ECG 信号侧错位检测 | JMIR 2021（DOI 10.2196/25347）；Physiol. Meas. 2024（DOI 10.1088/1361-6579/ad43ae）；系统综述 DOI 10.1016/j.jelectrocard.2020.08.013 | 深度学习/机器学习从 12 导联信号判断电极错位 | — | M4/M5 信号侧闭环验证 |

### 3.2 空白与机会

公开文献（OpenAlex 标题/摘要检索 "robot ECG electrode placement"）中**没有"机器人自动贴放 ECG 电极"的直接工作**：

- 现有研究分两类：**"看已经贴在哪"**（RGB-D + 检测，El Ghebouli/Bayer）与**"算应该贴哪"**（MRI/几何，Li/JBHI）；
- **执行环节（定位 → 到达 → 按压 → 自检 → 重贴）是空白**；
- 机器人相对人工的独特优势：可闭环自检、可自动重贴、可标准化。

### 3.3 对 pipeline 的六条设计原则（合并自 MD 总结）

| 编号 | 原则 | 落地位置 |
|---|---|---|
| **P1** | **学习层放在解剖 landmark，不放在电极 xyz**：检测器输出胸骨切迹/胸骨尖/锁骨/肋骨水平等 landmark，再由固定临床规则生成 V1–V6 | M3b |
| **P2** | **Surface/topology 约束一等原则**：规则的与学习的目标都必须投影到躯干表面 + 取法向，禁止自由 xyz | M1/M2/M3 |
| **P3** | **双胸廓坐标系**：仿真 GT 坐标系（信息丰富）vs 可感知坐标系（只用真实检测器可输出的 landmark）；M3 评估两者差距 | M0 定义 / M3 评估 |
| **P4** | **闭环验证与自动重贴**：按压后视觉重定位（仿真读位姿 + 真机 YOLO）与信号侧错位检测双通道 | M4 / M5 |
| **P5** | **参数化人体族**：体型/身高/胸廓厚度变化用于合成训练数据 + 泛化评估 + 可选体型模板迁移 | M0 / M3b / M4 |
| **P6** | **近距/多视角验证接口**：复用腕部相机做按压前/后确认；多视角 RGB-D 可达 2–6 mm 验证精度 | M2 / M4 |

---

## 4. Pipeline 总览

```
M0 场景·体型族·可达性
  └─ 仰卧姿态 + 床 + UR3 基座搜索 + 干电极末端 + 参数化体型族 + GT 定义协议
        │
M1 胸廓坐标系 + 规则目标 + 到达（GT 骨骼路径）
  └─ ChestFrame(GT) → V1–V6 规则（独立真值标定）→ 局部曲面拟合求表面点/法向 → 逐点 reach
        │
M2 RGB-D 深度融合（首版必做）
  └─ 目标投影 → 皮肤点径向修正 + 鲁棒 PCA 法向 → 解剖约束融合（越界回退）
        │
M3 视觉 landmark（3a 几何外推 → 3b 合成数据检测器）
  ├─ 3a: MediaPipe 33 点 → 可感知 ChestFrame → 规则 → V1–V6
  └─ 3b: Isaac 合成数据（体型族×姿态×光照×呼吸）→ landmark 检测器 → 同一套规则
        │
M4 多目标序列 + 按压 + 闭环验证 + 扰动评估
  └─ TSP 排序 → 安全面 → approach/press/retreat → 视觉自检 → 超差重贴 → 评估报告
        │
M5 迁移（Z1 Pro / 真机 / 信号侧闭环 / VLA 视觉伺服）
```

### 4.1 里程碑、交付物与验收

| 阶段 | 内容 | 交付物 | 验收标准（目标值，可调） |
|---|---|---|---|
| **M0** | 仰卧姿态资产（自定义 SkelAnimation 或根旋转）、床体、UR3 基座候选位姿搜索（左/右/头侧）、干电极末端（法兰+FT+弹簧电极）、体型族（≥5 配置）、GT 定义协议 | `task_manager/supine_pose.py`、`task_manager/ecg_scene.py`、`configs/ecg_scene.yaml`、`scripts/m0_check_ecg_scene.py`、`docs/ECG_M0_FINDINGS.md` | V1–V6 六点全部可达（IK 收敛、按压方向无碰撞）；至少 1 个基座位姿通过；间隙 ≥2 cm；体型族全部可加载 |
| **M1** | `ChestFrame`（GT 版）、V1–V6 规则（`configs/ecg_rules.yaml`）、局部 PCA 曲面拟合求表面点/法向、逐点 reach、**独立真值验证（25 躯干模型）**、基座候选×全目标联合搜索、接近方向倾角阶梯 | `target_localization/chest_frame.py`、`target_localization/ecg.py`、`task_manager/m1_demo.py`、`scripts/m1_ecg_reach.py`、`scripts/validate_rules_vs_torso_models.py`、`docs/ECG_GT_VALIDATION.md`、视频 | **PASS**（2026-09-19 第二轮）：基座 IK **6/6**、全部 tilt=0°；执行最小间隙 **3.87 cm**；822 帧；V6=真实腋中线 |
| **M2** | 目标投影到 RGB-D → 皮肤点（径向修正）+ 鲁棒 PCA 法向 + 解剖约束融合 + 越界/掠射回退 | `target_localization/fusion.py`、`task_manager/m2_pipeline.py`、`scripts/m2_ecg_depth.py` | **PASS**（2026-09-19 第二轮）：valid_rate 1.00（V6 掠射正当回退）；位置 1.12 mm；法向 0.89° |
| **M3a** | MediaPipe 33 点 → 可感知 ChestFrame（肩/髋/耳 + 人体测量比例）→ 规则 → V1–V6 | `perception/chest_frame_from_pose.py`、`scripts/m3a_ecg_pose.py`、`docs/ECG_M3A_FINDINGS.md` | **实测不可用**：俯视仰卧为 OOD（左右互换、手臂虚构、3D landmarks 失真），坐标系原点偏 101 mm、目标无法生成 |
| **M3b** | Isaac 合成数据集生成（尺度/位置/手臂姿态随机 + rig 关节投影标注）+ CNN landmark 检测器（P1）+ 逐像素提升 + 规则 + M2 融合 + 训练集标定（旋转/逐 landmark 偏移） | `scripts/gen_ecg_synth_dataset.py`、`scripts/train_chest_landmark.py`、`scripts/m3b_ecg_detector_eval.py`、`docs/ECG_M3B_FINDINGS.md` | **PASS**（v4，2026-09-19）：数据集重生成（手臂外展 60–90° 随机）；像素 0.13 px；**端到端 6.05 mm**（目标 ≤10 mm，最大 31.7 mm）；0/60 失败 |
| **M4** | TSP 多目标序列 + 胸部上方安全面 + approach/press/retreat + 力/压深限制 + 视觉自检 + 重贴（P4）+ 5 维度扰动评估（P5/P6） | `robot_controller/press_plan.py`、`task_manager/m4_demo.py`、`scripts/m4_ecg_place.py`、`scripts/m4_ecg_eval.py`、`docs/ECG_M4_FINDINGS.md`、视频 + JSON 报告 | **PASS**：一次成功 **6/6 = 100%**（目标 ≥90%）；到达误差 0.18–0.58 mm（≤1 cm）；力 0.60 N / 压深 4 mm **零违规**；执行最小间隙 **3.48 cm**；扰动评估最差均值 22.0 mm（患者 +2 cm），重贴路径为兜底 |
| **M5** | Z1 Pro 迁移（仅替换 `robot_controller` + 装配）；真机力控；ECG 信号侧错位检测（P4）；可选 VLA 视觉伺服 | 迁移文档 + 真机报告 | Z1 迁移后其余模块零改动；真机接触力达标率 |

---

## 5. 关键技术设计

### 5.1 双胸廓坐标系（P3）

**患者资产（2026-09-18 决定）**：使用官方**裸体**人体 `biped_demo`（Isaac 5.0，单一 16.8k 点网格 + 81 关节）。原 `M_Medical_01` 的衣物无法脱除（皮肤网格没有躯干表面），不满足"电极贴于裸露皮肤"的医学前提。详见 `ECG_M0_FINDINGS.md` §0。

**GT 坐标系（仿真）**：由 biped rig 关节构建
- `up = normalize(Neck1 − Spine3)`
- `lateral = normalize(L_Clavicle − R_Clavicle)`（患者左为正）
- `anterior = normalize(cross(lateral, up))`，实现时用朝向胸部外侧做符号校验
- origin = 双侧 `Clavicle` 中点（胸骨上端代理）
- **纵向参考：该 rig 没有乳腺/肋骨关节**，第 4/5 肋间水平必须由表面测量或学习模型给出（禁止固定值）；已发表的乳头/第 4 肋间统计（DOI 10.1097/00006534-200112000-00015）只作为总体先验保留
- 横向参考：胸骨中线 = 过 origin 的 v=0 平面；锁骨中线 = 单侧锁骨中点（`0.5 × (Clavicle + UpArm)`）的垂线；腋前线/腋中线由该高度躯干表面轮廓测量

**可感知坐标系（真实/感知）**：只用真实检测器可输出的 landmark
- 胸骨切迹（suprasternal notch）、胸骨尖/剑突、左右肩峰、髂嵴（JBHI 2024 路线）
- 肋间由**学习式 landmark 检测器**或表面测量给出（M3b 的核心任务）

**协议**：M1 用 GT 坐标系；M3a/M3b 输出可感知坐标系下的目标，同时报告与 GT 坐标系目标的差距——该差距是 sim2real 的核心风险指标。

**已知局限（必须写入文档）**：`biped_demo` 为平滑人体模型，无胸骨/乳头/肋骨的表面细节；仿真 GT 属于"资产内部一致定义"，非临床金标准。评估结论应表述为"与仿真 GT 的一致性"。此外该 rig 无 SkelRoot，关节读取走手动 FK 后备路径（已与 `UsdSkel.Cache` 对比验证，最大差 0.000 mm）。

### 5.2 V1–V6 规则（参数化，`configs/ecg_rules.yaml`）

在 ChestFrame 内定义（origin 在胸骨中线与第 4 肋间交点，u=纵向向上，v=横向向左，n=前向）：

| 电极 | u（肋间） | v（横向） | 临床对应 |
|---|---|---|---|
| V1 | 4th ICS | −parasternal_offset | 胸骨右缘 |
| V2 | 4th ICS | +parasternal_offset | 胸骨左缘 |
| V3 | (V2+V4)/2 | (V2+V4)/2 | 中点 |
| V4 | 5th ICS (= 4th − rib_spacing) | midclavicular | 左锁骨中线 |
| V5 | V4 同水平 | V4 + v4_to_v5 | 左腋前线 |
| V6 | V4 同水平 | V5 + v5_to_v6 | 左腋中线 |

默认参数（成人，可调）：

```yaml
parasternal_offset_m: 0.020   # V1/V2 距胸骨中线
rib_spacing_m: 0.020          # 第 4→5 肋间纵向间距
v4_to_v5_m: 0.035             # 锁骨中线 → 腋前线
v5_to_v6_m: 0.035             # 腋前线 → 腋中线
midclavicular_ratio: 0.5      # 锁骨中线 = ratio × 肩峰横坐标
```

优先使用解剖参考线（锁骨中线/腋前线/腋中线，由躯干宽度与体表 landmark 定义）；解剖线不可感知时回退到上述偏移量。

### 5.3 表面吸附与接近位姿（P2）

1. 规则目标先在 ChestFrame 内生成（三维空间中的名义点）；
2. **表面吸附**：从体外沿 `anterior`（或相机射线）与人体 mesh/深度点云求交 → 皮肤点 `p_s`；真机走 M2 的 `surface_point_from_depth`（复用甲状腺 `perception/depth.py`）；
3. **法向**：仿真取 mesh 法向；真机取窗口点云 PCA（大窗口 + 中值滤波，防肋骨带偏），并与 ChestFrame 先验做夹角约束（≤25°）；
4. **末端位姿**：`tool0` 目标 = `p_s` + 法向 × 电极长度；接近方向 = −法向（仰卧时近似世界 −Z）。

### 5.4 按压与力控

- 轨迹：`approach（沿法向，安全距离 5 cm 起）→ press（压深 2 mm / 力 0.5–2 N）→ hold（真机保持接触）→ retreat（回安全面）`；
- 仿真：读接触力（或压深代理），超过硬上限（3 N）立即回退；
- 真机（M5）：FT 传感器导纳控制，保持恒力；呼吸导致胸壁起伏时以力控维持接触；
- 每次按压后进入闭环验证（5.6）。

### 5.5 多目标序列与安全

- 序列：最近邻/TSP 排序（6 点规模，穷举最优即可）；
- 安全面：胸部上方 5 cm 的虚拟平面，点间移动沿该平面进行；
- 安全校验：复用甲状腺 `robot_controller/safety.py`（仰卧姿态下重算人体胶囊）+ 床体碰撞盒 + 执行期逐帧间隙校验；
- 目标钳制：规划失败沿接近方向回退（复用 `plan_reach` 的 backoff），绝不穿模。

### 5.6 闭环验证与重贴（P4）

```
按压完成
  ├─ 仿真：读取电极末端实际位姿 → 与意图目标比对
  ├─ 真机：固定 RGB-D（El Ghebouli 路线，YOLO + 深度）或腕部相机近距重定位
  ├─ 接触质量：法向夹角 / 压深 / 接触力
  └─ 判定：|Δp| ≤ 10 mm 且 夹角 ≤ 25° → 通过
             否则 → 抬起，用修正后的目标重贴（最多 2 次），仍失败则报告 FAIL 并跳过
真机追加（M5）：采一段 ECG → 信号侧错位检测（JMIR 2021）→ 作为最终质量门
```

### 5.7 评估协议与指标

**扰动维度**（对齐甲状腺 M4 的 7 扰动方法，扩展为 5 类）

| 维度 | 范围 |
|---|---|
| 体型 | ≥5 配置：身高/胸廓宽/胸廓厚非均匀缩放（含 BMI 近似变化） |
| 姿态 | 手臂位置（体侧/腹部）、头旋转、躯干轻微旋转 |
| 呼吸 | 吸气/呼气相位（胸壁 z 偏移 5–10 mm） |
| 相机 | 位置 ±5 cm、朝向 ±5° |
| 患者位置 | 平移 ±2 cm、偏航 ±5° |

**指标**

| 类别 | 指标 |
|---|---|
| 定位 | 各电极定位误差（cm，均值/最大）、肋间命中率（%） |
| 表面 | 法向夹角（°）、表面吸附残差（mm）、有效深度命中率（%） |
| 接触 | 压深（mm）、接触力（N）、力违规次数 |
| 安全 | 最小间隙（cm）、碰撞/穿模次数（必须为 0） |
| 闭环 | 一次成功率、重贴率、最终成功率 |
| 效率 | 全流程时长、单点时长 |
| 泛化 | 上述指标按扰动维度分组统计 |

**外部验证集（M3/M4 离线验证，2026-09-18 加入）**

- **PhysioNet/CinC Challenge 2007（Dalhousie）**：真实躯干 352 节点 + **120 个实测电极位置**（全部精确落在表面节点上），标准 12 导联为其中子集；许可 ODC-By 1.0。落地于 `assets/external/physionet_cinc2007/`，加载器 `roboecg/perception/external_torso.py`。
  - 用途：验证"电极定位/配准"（当作已贴电极恢复位置）与胸廓参考线测量；若确定 V1–V6 与电极编号的映射（查 Horáček Dalhousie 文献或一次人工标注并注明来源），可直接比对目标位置。
  - 限制：仅躯干、无骨架；映射未随文件发布。
- 已排查不可用：El Ghebouli 2025（联系作者）、Li 2025（UK Biobank 受控）、Isaac DH 数字人（Nucleus DNS 不可达且穿衣服）、openCARP（未确认）。
- 结论：**不存在公开的"完整裸体人体 + ECG 电极标注"模型**；仿真侧用 `biped_demo` 裸体资产 + 自建 GT（必须标注为自建），外部真实标注用上述 PhysioNet 躯干数据补足。

---

## 6. 项目结构（`RoboECG/`）

```
RoboECG/
├── configs/
│   ├── ecg_rules.yaml          # V1–V6 规则参数（5.2）
│   ├── ecg_scene.yaml          # 仰卧场景、基座、末端、相机
│   └── camera.yaml             # 复用甲状腺相机几何
├── roboecg/
│   ├── perception/
│   │   ├── isaac_skeleton.py   # 复用：读 GT 骨骼
│   │   ├── landmarks.py        # 复用 + ChestLandmarks
│   │   ├── depth.py            # 复用：皮肤点/法向/点云
│   │   ├── pose2d.py           # 复用：MediaPipe 3D 提升
│   │   ├── body_tracking.py    # 复用：子进程调用
│   │   ├── chest_frame_from_pose.py   # M3a
│   │   └── chest_detector.py          # M3b 学习式 landmark 检测
│   ├── target_localization/
│   │   ├── chest_frame.py      # 双坐标系定义（5.1）
│   │   ├── ecg.py              # V1–V6 规则生成（5.2）
│   │   └── fusion.py           # 表面吸附 + 解剖约束（5.3）
│   ├── coordinate_transform/   # 复用：frames.py, camera.py
│   ├── robot_controller/
│   │   ├── ur3_lula.py         # 复用
│   │   ├── reach_plan.py       # 复用（单点）
│   │   ├── press_plan.py       # 新增：多目标序列 + 按压
│   │   └── safety.py           # 复用（仰卧胶囊 + 床）
│   └── task_manager/
│       ├── supine_pose.py      # M0 仰卧姿态
│       ├── ecg_scene.py        # M0 场景装配 + 体型族
│       ├── scene_props.py      # 复用 + 电极末端
│       ├── overlay.py          # 复用：视频叠加
│       ├── m1_demo.py … m4_demo.py
│       └── m4_pipeline.py
├── scripts/
│   ├── env.sh / run_headless.sh       # 复用
│   ├── m0_check_ecg_scene.py
│   ├── m1_ecg_reach.py
│   ├── m2_ecg_depth.py
│   ├── m3a_ecg_pose.py
│   ├── gen_ecg_synth_dataset.py       # M3b 数据生成
│   ├── train_chest_landmark.py        # M3b 训练
│   ├── m3b_ecg_detector_eval.py
│   ├── m4_ecg_place.py
│   └── m4_ecg_eval.py
├── tests/                      # 纯逻辑单测（无 Isaac 依赖）
│   ├── test_chest_frame.py
│   ├── test_ecg_rules.py
│   ├── test_fusion.py
│   └── test_press_plan.py
├── docs/
│   ├── ECG_PIPELINE.md         # 本文件
│   └── ECG_M*_FINDINGS.md
├── assets/                     # 本地生成的 USD
└── runs/                       # 报告 / 视频 / 证据
```

---

## 7. 从甲状腺复现的复用清单

| 甲状腺文件 | ECG 用法 | 修改点 |
|---|---|---|
| `farus/coordinate_transform/frames.py` | 直接复制 | 无 |
| `farus/coordinate_transform/camera.py` | 直接复制 | 无 |
| `farus/perception/depth.py` | 直接复制 | 胸部窗口/鲁棒性参数可调 |
| `farus/perception/pose2d.py` | 直接复制 | 增加胸廓坐标系外推 |
| `farus/perception/isaac_skeleton.py` | 直接复制 | 无 |
| `farus/perception/body_tracking.py` | 直接复制 | 无 |
| `farus/robot_controller/ur3_lula.py` | 直接复制 | 无 |
| `farus/robot_controller/reach_plan.py` | 直接复制 | 供 `press_plan.py` 调用 |
| `farus/robot_controller/safety.py` | 复制 + 扩展 | 仰卧胶囊、床体碰撞 |
| `farus/task_manager/overlay.py` | 直接复制 | 无 |
| `farus/task_manager/scene_props.py` | 复制 + 扩展 | 增加干电极末端 |
| `farus/task_manager/m1_demo.py` | 结构参考 | 重写为仰卧 + 6 目标 |
| `farus/target_localization/fusion.py` | 约束思想复用 | 重写为多目标表面吸附 |
| `docs/SAFETY_FINDINGS.md`、M4 评估方法 | 方法论复用 | 扩展为 5 类扰动 |

复制文件在文件头注明来源（`farus_thyroid_isaac @ <commit>`），保持甲状腺项目为冻结交付物，避免回归风险。

---

## 8. 风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| UR3 臂展（500 mm）覆盖 V1（对侧胸骨旁）与 V6（腋中线） | 部分点不可达 | M0 基座位姿搜索（左/右/头侧三候选）；必要时提前引入 Z1 Pro（740 mm） |
| GT 坐标系基于蒙皮关节而非真实肋骨 | 仿真"精度"存在循环论证 | P3 双坐标系协议；M3 量化可感知坐标系 vs GT 的差距；文档声明局限 |
| MediaPipe 无胸骨/肋骨关键点 | M3a 误差可能 >2 cm | 这正是 M3b 的动机；两路线并行对比，以 M3b 为主力 |
| 肋骨使 PCA 法向偏斜（甲状腺已踩坑：13 cm 偏差） | 目标/接近方向被带偏 | 大窗口鲁棒拟合 + 与 ChestFrame 先验夹角 ≤25° + 越界回退 |
| 干电极仿真接触不稳定 | 按压阶段不可复现 | 首版压深代理 + 力读数记录；真机阶段导纳控制 |
| 合成数据检测器的 sim2real 差距 | M3b 迁移失败 | 域随机化（光照/材质/体型/相机）；保留 M3a 作为回退路径 |
| 呼吸导致按压瞬间目标偏移 | 接触力波动/贴偏 | 首版静态 + 扰动评估覆盖吸气/呼气相位；真机力控保持接触 |

---

## 9. 参考

**临床标准**
1. 12 导联电极位置与肢体导联定义（见输入文档 `ECG心电图Track.md`）。
2. Precordial lead displacement effect on ECG morphology, Med Biol Eng Comput 2013. DOI 10.1007/s11517-013-1115-9
3. Variability of precordial electrode placement, J Electrocardiol 1996. DOI 10.1016/s0022-0736(96)80080-x
4. Accuracy in ECG lead placement among technicians/nurses/physicians, Int J Clin Pract 2007. DOI 10.1111/j.1742-1241.2007.01390..x

**定位与生成**
5. Shashank et al., Precise Placement of Precordial Electrodes with ±0.5 cm Accuracy, 2017. DOI 10.1007/978-3-319-60483-1_55
6. Li et al., Personalized Topology-Informed Localization of Standard 12-Lead ECG Electrode Placement, Medical Image Analysis 2025. DOI 10.1016/j.media.2025.103472（arXiv:2408.13945）
7. Novel 3D Camera-Based ECG-Imaging System for Electrode Position Discovery and Heart-Torso Registration, IEEE JBHI 2024. DOI 10.1109/JBHI.2024.3520486
8. Torso geometry reconstruction and body surface electrode localization using 3D photography, J Electrocardiol 2017. DOI 10.1016/j.jelectrocard.2017.08.035

**验证与错位检测**
9. El Ghebouli et al., ECG electrode localization using 3D visual reconstruction, Front Physiol 2025. DOI 10.3389/fphys.2025.1504319
10. Bayer et al., ECG Electrode Localization: 3D DS Camera System, Sensors 2023. DOI 10.3390/s23125552
11. Reliable Deep Learning-Based Detection of Misplaced Chest Electrodes, JMIR 2021. DOI 10.2196/25347
12. A lightweight deep learning approach for detecting ECG lead misplacement, Physiol Meas 2024. DOI 10.1088/1361-6579/ad43ae
13. Machine learning for detecting electrode misplacement/interchanges: systematic review, J Electrocardiol 2020. DOI 10.1016/j.jelectrocard.2020.08.013

**机器人胸部任务**
14. Toward Lung Ultrasound Automation: Fully Autonomous Robotic Scans Along Intercostal Spaces, IEEE TMRB 2025. DOI 10.1109/TMRB.2025.3550663
15. Class-Aware Cartilage Segmentation for Autonomous US-CT Registration in Robotic Intercostal Ultrasound Imaging, IEEE TASE 2024. DOI 10.1109/TASE.2024.3411784
16. Tactile-Guided Robotic Ultrasound: Mapping Preplanned Scan Paths for Intercostal Imaging, IROS 2025. DOI 10.1109/IROS60139.2025.11246299

**上游复现**
17. Su et al., A fully autonomous robotic ultrasound system for thyroid scanning, Nat Commun 15, 4004, 2024. DOI 10.1038/s41467-024-48421-y
