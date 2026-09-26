# ECG M3a 结论：MediaPipe 几何外推在俯视仰卧场景失效

日期：2026-09-19
执行环境：Isaac Sim 6.0.1 + MediaPipe 1.0.1（独立 MediaPipe venv）
**状态：PARTIAL（路线不可用）**：坐标系轴向可用（≤1.6°），但原点误差 **101 mm**，且 V1–V6 目标生成直接失败。

## 1. 结果

| 指标 | 值 |
|---|---|
| MediaPipe 检测 | 33 关键点，可见度 0.91–1.00 |
| 坐标系轴向误差（修正后） | up **1.20°**、lateral **1.55°**、anterior **0.98°** |
| 坐标系原点误差 | **101.2 mm**（不可用） |
| 左右标签 | MediaPipe 输出**互换**（lateral 轴误差 160.6°），需按床位朝向修正 |
| V1–V6 生成 | **失败**：V5 吸附点 (u=−0.213, v=0.278) 落在躯干外 |

证据：`runs/m3a/m3a_report.json`、`pose_overlay.png`（骨架叠加）、`pose_keypoints.json`。

## 2. 失效原因（实测，非推测）

MediaPipe Pose 的先验是**直立人**，俯视仰卧视图属于分布外（OOD）：

| 现象 | 实测 | 说明 |
|---|---|---|
| 左右互换 | LEFT_SHOULDER 落在图像 v=479、RIGHT 在 v=310 | 图像上方=患者左侧（世界 +Y），标签系统性反了 |
| 手臂虚构 | 检测到腕部在图像上/下边缘（v=53 / 754），而实际手臂垂在体侧 | MediaPipe 把"手臂向两侧伸展"的直立先验套到仰卧图上 |
| 3D world landmarks 失真 | 髋宽 **0.132 m**（真值 ~0.25，偏窄 47%）；肩-髋高度差 0.085 m（仰卧应为 ~0） | 其内嵌 3D 姿态模型按站立姿态拟合 |
| 2D 肩点偏移 | 肩中点投影应在 v≈331，MediaPipe 给 v≈394.5（**偏 63 px ≈ 13 cm**） | 肩线被系统性放低 |

## 3. 已尝试的缓解（均不足以救回）

| 缓解 | 效果 |
|---|---|
| 按床位朝向修正左右标签 | lateral 轴 160.6° → 1.6°（有效） |
| 改用**逐像素深度提升**（不依赖 MediaPipe 的 3D 姿态） | 原点误差 300 mm → 94 mm；轴向 16° → 1° |
| 图像旋转 90°（头朝上）后重跑 | 左右仍互换、手臂仍虚构（无改善） |
| 关键点深度改用窗口内**最近深度**（避免取到桌面） | 无进一步改善（99 mm） |

残余 101 mm 原点误差来自 MediaPipe 的肩点系统性偏移（≈13 cm），无法用几何后处理消除。

## 4. 结论与对 pipeline 的影响

1. **M3a（MediaPipe 几何外推）在俯视仰卧场景不可用**：不是参数问题，是模型先验与视图不匹配。
2. 该结论**不否定几何外推本身**（甲状腺的坐姿场景可用），而是说明：**必须用与视图匹配的 landmark 检测器**。
3. 因此进入 **M3b：在 Isaac 合成数据上训练解剖 landmark 检测器**（pipeline 的 P1/P3 设计）——训练数据即来自本场景的俯视 RGB-D，天然规避 OOD 问题；同时用 P3 的双坐标系协议量化"可感知坐标系 vs GT 坐标系"的差距。
4. M3a 的两个副产品仍然有效：①"床位朝向已知 → 可修正左右"的工程手段；②逐像素深度提升在无遮挡视图下优于模型式提升（与甲状腺 M3 的结论互补：遮挡时用模型式，无遮挡时用逐像素）。

## 5. 证据索引

| 类型 | 路径 |
|---|---|
| 报告 | `runs/m3a/m3a_report.json` |
| 姿态叠加 | `runs/m3a/pose_overlay.png` |
| 输入图像 | `runs/m3a/pose_input.png` |
| 关键点 JSON | `runs/m3a/pose_keypoints.json` |
| 代码 | `roboecg/perception/chest_frame_from_pose.py`、`roboecg/task_manager/m3a_demo.py`、`scripts/m3a_ecg_pose.py` |

## 6. 复现命令

```bash
cd RoboECG
./scripts/run_headless.sh scripts/m3a_ecg_pose.py
```
