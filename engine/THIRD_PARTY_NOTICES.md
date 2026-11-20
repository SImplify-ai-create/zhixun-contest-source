# 第三方数据与组件声明（THIRD_PARTY_NOTICES）

本仓库在评测与演示中使用以下**公开基准数据集**。它们是比赛评测数据，不是"外部补充数据"。
逐文件的字节数与 sha256 见 `real_data/数据清单.md`；数据本身不入库，
放在与 `zhixun-engine` 同级的 `real_data\` 目录下。

| 数据集 | 用途 | 许可证 | 获取状态 |
|---|---|---|---|
| UCI Gas Sensor Array Drift（ID 224） | 分类主线、漂移监测与重训练（6 类 / 128 维 / 10 批） | CC BY 4.0 | 已取得并校验（13910 样本） |
| NASA C-MAPSS（FD001~FD004） | 回归主线、工况归一化、退化+工况切换下的漂移（14 字段 / 4 子集） | 美国政府作品，美国境内无版权（public domain），建议按学术惯例引用 | 已取得并校验 |
| XJTU-SY 滚动轴承全寿命 | 振动信号特征提取与剩余寿命（P2 优先级） | 以官方仓库声明为准（研究/教学用途），条款待核 | 数据包已到位但**仍是分卷 rar**（`real_data\XJTU-SY_Bearing_Datasets\Data\*.part01~06.rar`，4.37 GB，未解压）；官方口径已核对（25.6 kHz / 32768 点 × 2 通道 / 1 min 间隔 / 三工况 2100-2250-2400 r/min 对应 Bearing 1_x-2_x-3_x）；解压后先按官方表3 逐轴承核对 CSV 数量与采样点数再接入 |

## 引用

**UCI 气体传感器阵列漂移**

> Vergara, A., Vembu, S., Ayhan, T., Ryan, M. A., Homer, M. L., & Huerta, R. (2012).
> Chemical gas sensor drift compensation using classifier ensembles.
> *Sensors and Actuators B: Chemical*, 166–167, 320–329.
> 数据集：UCI Machine Learning Repository, ID 224. https://archive.ics.uci.edu/dataset/224/

**NASA C-MAPSS**

> Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).
> Damage propagation modeling for aircraft engine run-to-failure simulation.
> *Proceedings of the 1st International Conference on Prognostics and Health Management (PHM)*.
> 数据由 NASA Prognostics Center of Excellence (PCoE) 发布。

**XJTU-SY**

> Wang, B., Lei, Y., Li, N., & Li, N. (2020).
> A hybrid prognostics approach for estimating remaining useful life of rolling element bearings.
> *IEEE Transactions on Reliability*, 69(1), 401–412.

## 使用边界

1. 三套数据均为公开可引用数据集，**原先针对企业数据的保密红线不适用于它们**；
   但对外材料引用时必须标注来源与许可证（本文件即为此用途）。
2. `data\factory` 与 `real-j820189-*` 属**企业数据**，按企业数据严格隔离：
   不进入任何公开产物、不与本文件的公开数据集混跑、不写入 `real_data\`。
3. 结论口径：只用这三套公开数据主张"同数据同预算下的臂间相对优势"，不主张 SOTA。
4. 合成流（`src/data.py`）仅作内部冒烟/CI 夹具，标注 `INTERNAL FIXTURE ONLY`，
   不再作为对外证据；历史基于合成流的量化结论需逐条标注"历史合成流结果，仅内部追溯"。
