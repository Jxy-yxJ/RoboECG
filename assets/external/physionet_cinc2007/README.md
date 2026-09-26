# 外部验证数据：PhysioNet/CinC Challenge 2007（Dalhousie 躯干 + 120 电极）

## 这是什么

真实人体躯干表面与**实测电极位置**的公开数据，用于验证 ECG 电极定位算法（本项目 M3/M4 的离线验证集）。

| 文件 | 内容 |
|---|---|
| `case0003_b352.pts` | 躯干表面 **352 个节点** 的 3D 坐标（mm） |
| `case0003_b120.pts` | **120 个 BSPM 电极** 的 3D 坐标（mm）；实测每个电极都精确落在躯干节点上（0.00 mm） |
| `Horacek_torso_120_leads.pdf` | 120 导联在躯干展开图上的编号布局 |
| `Horacek_torso_352_nodes.pdf` | 352 节点布局 |

数据来源（可复现）：

- 数据集主页：https://physionet.org/content/challenge-2007/1.0.0/
- 原始文件：`https://physionet.org/files/challenge-2007/1.0.0/data/case0003_b120.pts` 等
- 同目录另有 `case0003_dat.mat`：`bspmdata.potvals` 为 (352, 1095) 的体表电位（uV），`fids` 为心搏时间基准点
- 案例 3 还提供心脏/肺/躯干几何（`case0003_h1100.pts/.fac`、`case0003-tri/`）

## 许可

**Open Data Commons Attribution License v1.0 (ODC-By 1.0)**（PhysioNet 页面标注 "License (for files)"）。
可自由使用与再分发，**必须署名**。引用时请注明：

> PhysioNet/CinC Challenge 2007 data (Dalhousie torso with 120 leads), https://physionet.org/content/challenge-2007/1.0.0/, ODC-By 1.0.
> 相关文献：Horáček et al., Body surface potential mapping of ST segment changes in acute myocardial infarction, Circulation 87(3):773, 1993 (DOI 10.1161/01.CIR.87.3.773).

## 用途（本项目的验证协议）

1. **电极定位/配准验证**：把 120 个电极当作"已贴好的电极"，验证感知与配准流程能否在真实躯干表面上恢复其位置（对应 El Ghebouli 2025 / Bayer 2023 的任务）。
2. **目标位置验证**：PhysioNet 页面明确说明标准 12 导联是 120 个电极的子集。若能从 Dalhousie 布局约定（Horáček 文献）确定 V1–V6 对应的电极编号，即可把本项目生成的 V1–V6 与该真实标注比对。
3. **胸廓坐标系验证**：在真实躯干几何上检验胸骨中线/锁骨中线/腋前线等参考线的测量方法。

## 已知限制（如实说明）

- 只有**躯干**：无头/手臂/腿，无骨架、无姿态 → 不能直接用于机器人场景，只用于离线定位验证（或作为静态躯干接入仿真）。
- `.pts` 只给点，不含三角网格连接；需要时可重建表面（如 ball-pivoting / Delaunay）。
- 坐标轴约定需在导入时核验（Dalhousie 惯例通常为 x=左右、y=前后、z=上下，单位 mm）。
- **V1–V6 与电极编号的对应关系未包含在本目录文件中**：布局 PDF 只给 120 导联编号网格；标准导联映射需查 Horáček 的 Dalhousie 文献，或做一次人工标注（必须记录为"人工标注"来源）。
- 这是 2007 年挑战赛的单个受试者几何（case 3），不是参数化人群。

## 其他候选（已排查）

| 资源 | 结论 |
|---|---|
| El Ghebouli et al. 2025（Liryc，7 名志愿者 + 幻影，118 电极 + ETS 真值） | 数据"included in the article/supplementary material, further inquiries to the corresponding author"——**不公开**，可联系作者 |
| Li et al. 2025（MedIA，10 电极 + 心脏 MRI） | 代码公开（github.com/lileitech/12lead_ECG_electrode_localizer），数据为 UK Biobank——**受控访问** |
| openCARP 躯干/ECG 示例 | 未确认（GitHub API 需鉴权），待查 |
| Isaac Sim DH 数字人（裸体可能性） | 衣服/配件在 `omniverse://avatar.ov.nvidia.com`，本机 **DNS 不可达**；且角色本身穿衣服，未验证衣服下身体是否完整 |
| MakeHuman（CC0） | 可生成裸体参数化人体，但**无电极标注**，标注需自行按临床规则定义 |
