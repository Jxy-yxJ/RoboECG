# ECG M0 结论：裸体仰卧场景、胸廓 landmark 审计与可达性

日期：2026-09-18
执行环境：Isaac Sim 6.0.1（`$ISAACSIM_ENV`），RTX 4060 Laptop 8GB，驱动 595.91.07
患者资产：**`biped_demo`（官方裸体人体，Isaac 5.0 资产，6.0 可用）**；单一 16,782 点网格 + 81 关节骨架

## 0. 关键决策：患者资产切换为裸体模型

ECG 电极必须贴在**裸露皮肤**上。首版使用的 `M_Medical_01` 角色穿 lab coat + scrub shirt，实测发现：

- 隐藏衣物后**躯干是空的**（该资产的皮肤网格只做了头/颈/手臂/腿，胸廓表面本来就是衬衫网格）；
- 因此 `M_Medical_01` **不能**用于"贴电极"的医学仿真。

候选评估：
- Isaac 6.0 全部官方角色（Medical/Police/Construction/Business）均穿衣服；
- `DH_Characters`（数字人）依赖 `omniverse://avatar.ov.nvidia.com`，本机无法加载（加载挂起）；
- **`biped_demo`**（Isaac 5.0）：单一裸体网格、81 关节骨架、无任何衣物 → **选定**。

代价（已记录）：该网格是平滑人体模型，**没有胸骨/乳头/肋骨的表面细节**；肋间水平必须由测量或学习模型给出（这正是 pipeline 的设计原则 P1/P3）。

## 1. 结论总览

| 检查 | 脚本 | 结果 | 证据 |
|---|---|---|---|
| 裸体仰卧场景装配（床/患者/UR3/电极末端/相机） | `scripts/m0_check_ecg_scene.py` | **PASS** | `runs/m0/m0_report.json`、截图 |
| rig 朝向实测与仰卧放置 | `scripts/m0_debug_biped_pose.py`、`m0_debug_supine.py` | **PASS** | 本文 §2 |
| 手动 FK 正确性（无 SkelRoot 资产） | `/tmp` 验证脚本 | **PASS**：与 `UsdSkel.Cache` 最大差 **0.000 mm**（101 关节） | 本文 §6 |
| 胸廓 landmark 实测 | `m0_check_ecg_scene.py` | **PASS**（带来源与 caveat） | `m0_report.json → landmark_audit` |
| 胸廓坐标系 | 同上 | **PASS** | `m0_report.json → chest_frame` |
| 胸廓表面（资产网格） | `--surface mesh` | 16/16 网格单元 | `m0_report.json → surface_grid` |
| 胸廓表面（RGB-D 俯视） | 默认路径 | **16/16 单元，85,087 点** | 同上 |
| 相机位姿对比 | 斜视 vs 俯视 | **斜视 9/16 → 俯视 16/16** | 本文 §5 |
| 规则 provenance 校验 | `roboecg/target_localization/ecg_rules.py` | **PASS**（8 条，无手填医学常数） | `m0_report.json → rules_provenance` |
| UR3 基座可达性 | `m0_check_ecg_scene.py` | **PASS**（39/135 候选全 16 目标可达，间隙 2.4 cm） | `m0_report.json → reachability` |
| 体型族（0.90–1.10） | `scripts/m0_body_family.py` | **PARTIAL**：IK 5/5 档通过；间隙 3/5 档达标 | `runs/m0/m0_body_family.json`、本文 §7 |
| 纯逻辑单测 | `scripts/run_tests.sh` | **12 passed** | `tests/` |

## 2. rig 朝向与仰卧放置

`biped_demo` 站立 T-pose 实测：身体长轴 +Z、患者左侧 +X、面朝 **-Y**（与 `M_Medical_01` 同约定）。仰卧旋转沿用 `Rz(90) @ Rx(-90)`：头朝 -X、胸朝 +Z、患者左侧朝 +Y（机器人侧）。

手臂姿态（rig 特定，经 `m0_debug_biped_pose.py` 实测轴搜索确定）：

| rig | 关节 | 局部轴 | 角度 | 效果（实测） |
|---|---|---|---|---|
| biped_demo | `L_UpArm` / `R_UpArm` | Z | **+90° / −90°** | 手从 x=0.72 m（T-pose）收到 x=0.19 m（体侧） |
| M_Medical_01（备用） | `L/R_Upperarm` | Z | ∓75° | 甲状腺复现沿用值 |

放置后：`human_root_translate=[1.05, 0, 0.849]`；桌面 z=0.70；渲染确认手臂在体侧、胸廓朝上、背部贴桌。

## 3. 胸廓 landmark 实测（asset measurement，非临床统计）

`m0_report.json → landmark_audit`（biped_demo，scale=1.0）：

| 量 | 值 (m) | 说明 |
|---|---|---|
| 锁骨长（L/R） | 0.1850 / 0.1850 | `Clavicle`（胸锁端）→ `UpArm`（肩端） |
| 肩宽（肩关节间距） | 0.3934 | `L_UpArm` ↔ `R_UpArm` |
| 胸锁端间距 | **0.0413** | 比 `M_Medical_01` 的 0.122 更接近真实胸锁关节间距（~0.06–0.07） |
| 颈底→上胸 | 0.2112 | `Neck1` ↔ `Chest` |
| 下脊柱→上胸 | 0.2254 | `Spine1` ↔ `Chest` |
| 胸→骨盆 | 0.2228 | `Spine3` ↔ `Pelvis` |

**Caveat**：该 rig **没有乳腺/肋骨关节**（只有 Spine1/2/3、Chest、Neck1/2、Clavicle、UpArm）。因此第 4 肋间水平在仿真中也不存在"免费 GT"，必须由测量或学习模型给出——这与 pipeline 的设计原则一致，也是相对 `M_Medical_01` 更诚实的设置。

## 4. 胸廓坐标系

| 轴 | 定义 | 来源 |
|---|---|---|
| up | `Spine3 → Neck1` | asset_measurement |
| lateral | `R_Clavicle → L_Clavicle`（对 up 正交化） | asset_measurement |
| anterior | `lateral × up`，符号由 hint（仰卧时世界 +Z）确定 | clinical_definition |
| origin | 双侧 `Clavicle` 中点（胸骨上端代理） | asset_measurement |

实测：origin=`[-0.4134, 0, 0.8709]`，anterior=`[-0.105, 0, 0.994]`。

## 5. 胸廓表面测量：网格 vs RGB-D，以及相机位姿实验

在 (u,v) 4×4 网格上取每格最前点并做 PCA 法向拟合：

| 来源 | 点数 | 覆盖 | 法向 vs 全局 anterior 均值/最大 |
|---|---|---|---|
| 资产网格（rest pose） | 16,782 | 16/16 | 18.6° / 57.4° |
| RGB-D **斜视**（患者右侧） | 82,316 | **9/16** | 24.2° / 55.5° |
| RGB-D **俯视**（胸部正上方） | 85,087 | **16/16** | **22.1° / 62.3°** |

**相机位姿实验（关键发现）**：单目斜视相机看不到左侧胸壁（V5/V6 区域，每格 0–6 点）；改为胸部正上方俯视后 16/16 全覆盖。**M2 感知相机默认俯视**已固化为场景配置。

**法向发现（影响 M2 设计）**：胸廓表面法向相对单一全局前向最大偏 62°（侧胸 v=0.172 格），胸骨中线附近仅 0.7–6°。甲状腺复现的"实测法向与先验夹角 ≤25°"约束来自颈部，**不能直接用于胸部**；M2 必须改为**局部先验**（按 (u,v) 给期望法向或局部曲面模型），否则 V5/V6 的法向会被系统性拒绝。

**相机内参修正**：原相机只设水平 aperture（20.955 mm），USD 默认垂直 aperture 15.2908 mm，在 16:9 下垂直 FOV≈72°，而 pinhole 模型（fx=fy）假设 ≈59°，会导致点云纵向拉伸。已在 `add_camera` 中设 `vertical_aperture = horizontal / aspect`，保证 fx=fy。

## 6. 可达性搜索

方法：以胸廓坐标系原点为基准搜索 135 个基座位姿；对 16 个已测表面目标做带姿态 IK（`tool0` 指向 `-anterior`，电极偏移 0.088 m），检查人体胶囊间隙（rig 自适应关节名）+ 桌面碰撞盒。仅检查目标位姿（全轨迹规划属 M1）。

| 指标 | 值 |
|---|---|
| IK 全 16 目标通过 | **39/135 候选** |
| 最佳基座 | `[-0.263, 0.45, 1.021]`，yaw 180° |
| 最小人体间隙 | **0.0241 m**（≥ 0.02 m 余量，达标） |
| 桌面间隙 | 0.268 m |
| `base_link_local` 校验 | 平移 0、旋转 0° |

**含义**：UR3（500 mm 臂展）在"患者左侧、基座高于桌面"的布局下可以覆盖全部胸廓目标，间隙 2.4 cm 达标；臂展风险不成立。

## 7. 体型族（uniform scale）

`runs/m0/m0_body_family.json`，对 `biped_demo` 做 0.90–1.10 均匀缩放，每档重测 landmark、表面与可达性：

| scale | 身高 (m) | 肩宽 (m) | 表面覆盖 | IK 通过候选 | 最小间隙 (m) | 间隙达标 |
|---|---|---|---|---|---|---|
| 0.90 | 1.638 | 0.354 | 16/16 | 51/135 | **0.0230** | 达标 |
| 0.95 | 1.729 | 0.374 | 16/16 | 42/135 | **0.0210** | 达标 |
| 1.00 | 1.820 | 0.393 | 16/16 | 39/135 | 0.0211 | 达标 |
| 1.05 | 1.911 | 0.413 | 16/16 | 39/135 | 0.0212 | 达标 |
| 1.10 | 2.002 | 0.433 | 16/16 | 39/135 | 0.0214 | 达标 |

**结论**：IK 可达性 5/5 档通过；早期碰撞胶囊半径是固定工程值（0.14/0.09/0.06 m），
不随体型缩放，使小体型（0.90/0.95）的最小间隙低于 2 cm 余量。已修正为按实测体型缩放
（`scale = |Spine3−骨盆| / 0.2228 m`，见 `robot_controller/safety.py`），修正后 5/5 档全部达标
（最小间隙 21–23 mm）。

## 8. 规则与 provenance

`configs/ecg_rules.yaml` 加载时强校验：每个解剖量必须带 provenance；`measured_or_learned` 不允许携带固定值；`published_statistic/regression` 必须有引用。本次 8 条全部通过：

- 第 4 肋间水平：`published_statistic`，乳头中心位于第 4 肋间 75%、第 5 肋间 23%（n=100 男性），DOI 10.1097/00006534-200112000-00015；**由于 biped rig 无乳腺关节，该统计量只作为总体先验保留，仿真目标必须由测量/学习给出**。
- 第 4→5 肋间距：`measured_or_learned`（**无固定值**）；依据：肋间距与身高/BMI/年龄无关（r=0.087），需直接测量，DOI 10.1002/micr.22238。
- 胸骨缘 / 胸骨旁偏移：`measured_or_learned`（禁止固定偏移）。
- 锁骨中线 / 腋前线 / 腋中线：`clinical_definition` + 资产实测 landmark。
- 接触压力：`published_statistic`，15 mmHg 下干电极信号质量与 Ag/AgCl 相当，DOI 10.3390/s20216233。
- 按压力目标：`engineering` 导出值 = 2000 Pa × 3×10⁻⁴ m² = **0.6 N**。

## 9. 过程中修复的工程问题（可复现的踩坑记录）

| 问题 | 现象 | 修复 |
|---|---|---|
| `M_Medical_01` 无裸胸 | 隐藏衣物后躯干为空 | 切换 `biped_demo` 裸体资产 |
| `biped_demo` 无 SkelRoot | `UsdSkel.Cache.Populate` 传入无效 root，脚本静默结束 | 新增**手动 FK** 后备路径（父链由关节路径名推导），与 Cache 对比 **0.000 mm** |
| `Skeleton.GetJointParentsAttr().Get()` | 属性未授权时调用 `Get()` 触发硬崩 | 一律从关节路径名推导父子关系 |
| `UsdSkel.BindingAPI(prim)` 对象构造 | 对未应用该 API 的 prim 构造会崩 | 改用原始关系 `skel:animationSource` 读写 |
| `Gf.Matrix4d.SetTransform` | Boost.Python 参数不匹配 | 改用 `Gf.Transform` + `Gf.Rotation` 组合 |
| Python 异常不可见 | 异常只写 kit 日志，stdout 无输出 | 调试时同时检查 `isaacsim/kit/logs/.../kit_*.log` |
| `body_capsules` 关节名硬编码 | 换 rig 后 KeyError | 关节名候选列表，双 rig 兼容 |
| 相机垂直 aperture 缺失 | fx≠fy，点云纵向拉伸 | `vertical_aperture = horizontal / aspect` |

## 9.5 外部验证数据（新增，用于 M3/M4 离线验证）

**PhysioNet/CinC Challenge 2007（Dalhousie）**：真实人体躯干表面 + 实测电极位置。

| 项 | 值 |
|---|---|
| 躯干表面节点 | 352 个（3D，mm） |
| 电极位置 | **120 个**，实测全部精确落在躯干节点上（0.00 mm） |
| 标准 12 导联 | PhysioNet 页面明确为标准导联是 120 个的子集 |
| 附带数据 | `case0003_dat.mat` 体表电位 (352×1095, uV)；case 3 心脏/肺/躯干几何 |
| 许可 | **ODC-By 1.0**（署名即可） |
| 本机落地 | `assets/external/physionet_cinc2007/`（含 README、.pts、布局 PDF）；加载器 `roboecg/perception/external_torso.py`；单测 `tests/test_external_torso.py` |

**限制（如实）**：仅躯干（无头/臂/腿、无骨架）；`.pts` 只有点不含网格；**V1–V6 与电极编号的映射不在文件里**（需查 Horáček Dalhousie 文献或做一次人工标注并注明来源）；单受试者。

**DH 数字人（裸体可能性）排查结论**：DH 角色的衣服/鞋/配件均在 `omniverse://avatar.ov.nvidia.com`，本机 **DNS 不可解析**（网络层不可达），且 DH 本身穿 workwear/casual；衣服下身体是否完整未验证。**当前环境不可用**；若在可访问 Nucleus 的网络/账号下可再试。

**其他候选排查**：El Ghebouli 2025 数据不公开（联系作者）；Li 2025 数据受 UK Biobank 限制；openCARP 未确认；MakeHuman（CC0）可生成裸体但无电极标注。

## 10. 证据索引

| 类型 | 路径 |
|---|---|
| 主报告 | `runs/m0/m0_report.json` |
| 体型族 | `runs/m0/m0_body_family.json` |
| 截图 | `runs/m0/third_view.png`（患者+UR3）、`runs/m0/chest_closeup.png`（胸廓标记）、`runs/m0/depth.png` |
| 资产诊断 | `scripts/m0_debug_assets.py`（任意人体资产的网格/骨架/包围盒） |
| 姿态轴搜索 | `scripts/m0_debug_biped_pose.py` |
| 表面对比 | `scripts/m0_debug_surface.py` |
| 单测 | `tests/test_chest_frame.py`、`tests/test_ecg_rules.py` |

## 11. 复现命令

```bash
cd RoboECG
./scripts/run_tests.sh
./scripts/run_headless.sh scripts/m0_debug_biped_pose.py              # 姿态轴搜索
./scripts/run_headless.sh scripts/m0_check_ecg_scene.py --surface mesh --no-render
./scripts/run_headless.sh scripts/m0_check_ecg_scene.py               # RGB-D + 截图
./scripts/run_headless.sh scripts/m0_body_family.py
```

## 12. 局限与下一步

**局限**
- `biped_demo` 为平滑人体模型：无胸骨/乳头/肋骨表面细节，M3b 学习式 landmark 检测器在仿真中可依赖的**表面线索有限**（关节 GT 可用，但真实感知需估计）；这正是 M3b 要解决的问题。
- 仰卧为刚体近似（未做脊柱放平）。
- 碰撞胶囊半径为固定工程值 → 小体型间隙偏保守（§7）。
- 可达性仅检查目标位姿，未做全轨迹碰撞规划（M1）。
- 体型族为均匀缩放。
- 深度表面与网格表面存在 ~1 cm 量级 n 值差（平滑网格 vs 渲染皮肤），M2 需量化。

**下一步（M1）**
1. V1–V6 规则生成：纵向（第 4/5 肋间）由**测量或学习**给出（禁止固定偏移）；横向（胸骨缘、锁骨中线、腋前线/中线）由表面几何测量；先实现网格/深度测量版本，M3b 换学习版。
2. 用 `plan_reach(..., ground_box=...)` 做全轨迹规划与执行期间隙校验。
3. 碰撞胶囊随体型缩放，解决小体型间隙问题。
4. 录制定点到达视频（对齐甲状腺 M1–M4 证据格式）。
