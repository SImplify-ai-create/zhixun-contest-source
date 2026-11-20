# zhixun-contest-source

本仓库包含两个顶层目录：

## submission/

实际提交镜像所用的源码。

- 镜像名：`zhixun-harness:v3-repick`
- 镜像 tar 的 sha256：`9D8B4870439C2666403ACF393EF74FA97EDA868CBFD109CFA169B84FF6765FB1`
- 镜像内 `/app` 与 `submission/app/` 下的 10 个 `.py` 文件逐字节一致。
- 目录内容：`app/`、`tests/`、`manifest.json`、`P0_FREEZE.json` 及随包说明文档。

## engine/

方法学与实验引擎的参考实现，**不是镜像内容**。

- 目录内容：`src/`、`docs/`、`tests/`、`config.json`、`README.md`、`THIRD_PARTY_NOTICES.md`、`requirements.txt`。
- 不含原始数据、运行记录、模型权重与私有配置；实验记录见公开实验记录页。
