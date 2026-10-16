# Qwen3.8-Flash-Next 在海光 K100-AI（gfx928）上的 INT8 部署与优化

> 把官方 **Qwen3.8-Flash-Next INT8 (W8A8)** 权重 + 官方 vLLM 镜像，在一张**厂商未适配**的国产卡（海光 K100-AI / gfx928）上跑通并优化到可用：4 卡 TP4，**单流 ~31 tok/s、20 并发聚合 ~175 tok/s**，262K 原生上下文、MTP 投机解码、工具调用与思维链分离、前缀缓存，全部实测可复现。

## 先准备这两样（镜像 + 模型）

**只此两样**，都从各自官方渠道拿，不需要本项目的任何账号或凭据：

| 资源 | 地址 | 体积 | 校验值 |
|---|---|---|---|
| 容器镜像（海光官方） | `harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.29.0-ubuntu22.04-dtk26.04-py3.10-20260831-qwen3.8flashnext` | 42.9 GB（`docker save` 后 8.2 GiB） | 清单 digest `sha256:636005656b97927f93e04f03258e8380d650448e9a263e54929e281c47315670`；tar MD5 `a1aab76a49f1af869742ce3bf71db0e9` |
| 模型权重（ModelScope，海光发布） | <https://modelscope.cn/models/hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8> | 170.8 GiB（145 文件） | 逐文件 sha256，用 `tools/verify_model_download.py` 一键比对 |

```bash
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.29.0-ubuntu22.04-dtk26.04-py3.10-20260831-qwen3.8flashnext   # 镜像（实测匿名可拉）
pip install modelscope                                                                                                          # 模型下载工具
modelscope download --model hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8 --local_dir /data2/qwen38-int8-w8a8
```

两样备齐后，**本仓库本身不含镜像与权重本体**（加起来 0.5 MB 的文本补丁 + 脚本）；完整步骤见[第四章 快速开始](#四快速开始)，目标机不能联网时的转存办法也在那里。

---

## 一、目标机器

| 项目 | 说明 |
|---|---|
| 加速卡 | 海光 **K100-AI**（`gfx928`，深算二号血统，单卡 64 GB HBM） |
| 本方案配置 | **4 卡 TP4**（原方案为 8 卡，本机另 4 卡跑其他在线服务，全程未受影响） |
| 互联 | 板内 xGMI |
| 宿主 | Linux（Kylin V10 实测），驱动/用户态来自 DTK 26.04 |
| 磁盘 | 权重 ~171 GiB + 编译缓存若干 GiB，建议预留 300 GiB 以上 |

> 卡不是厂商适配列表里的型号（官方 vLLM/算子包只标 BW1000 系列），所以本项目本质上是一份"**让未适配卡跑起来**"的工程记录。

## 二、目标环境（镜像内实测版本）

| 组件 | 版本 |
|---|---|
| 镜像 | `harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.29.0-ubuntu22.04-dtk26.04-py3.10-20260831-qwen3.8flashnext` |
| 镜像体积 | 42.9 GB（`docker save` 后 tar 8,818,734,592 字节，MD5 `a1aab76a49f1af869742ce3bf71db0e9`） |
| DTK | 26.04 |
| vLLM | 0.29.0（+ das 插件） |
| PyTorch | 2.11.0 |
| Triton | 3.6.0 |
| Python | 3.10.12 |
| 系统 | Ubuntu 22.04（镜像内） |

宿主需要把 `/opt/hyhal` 只读挂进容器，并在容器内 `source /opt/dtk/env.sh && source /opt/hyhal/env.sh`（起服脚本里已经写好）。

## 三、目标模型

| 项目 | 说明 |
|---|---|
| 模型 | Qwen3.8-Flash-Next（125B MoE，激活约 6B，含 51B N-gram/PLE 表） |
| 权重 | **INT8 W8A8**，compressed-tensors 格式 |
| 架构标识 | `Qwen4ExpForConditionalGeneration`（`model_type: qwen4_exp`） |
| 原生上下文 | 262,144 token |

### 模型下载地址（ModelScope）

```
https://modelscope.cn/models/hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8
```

- 体积：**183,374,474,955 字节（170.8 GiB）**，共 **145 个文件**（131 个 safetensors 分片 + 14 个小文件）。
- 命令行下载：

```bash
modelscope download \
  --model hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8 \
  --local_dir /data2/qwen38-int8-w8a8
```

> **务必逐文件校验 sha256**。170 GiB 该断点续传就续传（不用才不现实），但**续传之后一定要比对官方 sha256**：尺寸只说明"下够了字节"，不说明"下对了内容"。本项目就踩过这个坑——续传写错偏移，文件尺寸全对、safetensors 头也能解析，内容却是坏的（详见《优化历程》第 1 节）。

## 四、快速开始

### 0. 机器前置条件

| 前置 | 说明 |
|---|---|
| 驱动 / 用户态 | 装好 DTK 26.04（提供 `/dev/kfd`、`/dev/dri`、`/dev/mkfd` 与 **`/opt/hyhal`**；起服脚本会把它只读挂进容器） |
| docker | 起服脚本默认用标准 socket `unix:///var/run/docker.sock`；不是这个位置就 `DOCKER_SOCK=...` |
| 显卡工具 | 宿主有 `hy-smi`（仅用于打印卡位，缺了不影响起服） |
| Python | 仅在跑校验/压测脚本时需要（系统自带 python3 即可，无第三方依赖） |
| 下载工具 | 拉模型需要 ModelScope CLI：`pip install modelscope`（国内可用 `-i https://mirrors.aliyun.com/pypi/simple/` 加速） |
| 磁盘 | 权重 171 GiB + 编译缓存（首次约 20~30 GiB）+ 富余，建议 ≥ 300 GiB |

### 1. 准备镜像

```bash
# 方式一（推荐，最简单）：直接从海光官方镜像仓库拉取
#   已验证：该项目**匿名可拉**，不需要账号
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.29.0-ubuntu22.04-dtk26.04-py3.10-20260831-qwen3.8flashnext

# 方式二：目标机不能联网时，在一台能上网的机器上转存（这正是本项目的做法）
#   2.1 在有网的机器上
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.29.0-ubuntu22.04-dtk26.04-py3.10-20260831-qwen3.8flashnext
docker save harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm0.29.0-ubuntu22.04-dtk26.04-py3.10-20260831-qwen3.8flashnext \
  -o vllm0.29.0-qwen3.8flashnext.tar
#   42.9 GB 的镜像 docker save 后是 8,818,734,592 字节（约 8.2 GiB），层是压缩存的
#   2.2 传到目标机（任意方式：U 盘 / scp / 共享目录）
#   2.3 在目标机上导入
docker load -i vllm0.29.0-qwen3.8flashnext.tar
#   校验：文件 MD5 应为 a1aab76a49f1af869742ce3bf71db0e9，字节数 8,818,734,592

# 方式三：用本仓库的 Dockerfile 构建一个"已打补丁"的镜像
docker build -t k100-qwen38-int8:patched .
# 之后起服时指定：IMG=k100-qwen38-int8:patched bash /data2/serve_cg_oai.sh
```

> **镜像校验值**（用来确认你拿到的是本项目验证过的那个版本）：
>
> ```
> 清单 digest : sha256:636005656b97927f93e04f03258e8380d650448e9a263e54929e281c47315670
> tar  MD5    : a1aab76a49f1af869742ce3bf71db0e9   （8,818,734,592 字节）
> ```
>
> `docker images --digests` 或 `docker inspect` 可对上前者。

> **本仓库不含镜像本体。** 离线 tar 8.2 GiB，无法进 git；仓库里全部是文本补丁、配置与脚本，加起来 0.5 MB 左右。
>
> 镜像有两条路：**方式一/二是"官方镜像 + 运行时挂载补丁"**（仓库默认姿势，镜像保持原厂，升级回滚干净）；**方式三是"把补丁烤进镜像"**（只在官方镜像之上加一层不到 1 MB 的层，构建和推送到镜像仓库都很快，基础层若已存在则复用）。两种方式跑起来行为一致。

本项目**不改镜像**：所有修复都以补丁文件的形式在起服时挂载进容器，镜像保持官方原样，升级/回滚都干净。

### 2. 准备模型

```bash
# 先装下载工具（只需一次）
pip install modelscope          # 国内可用 -i https://mirrors.aliyun.com/pypi/simple/ 加速

# 再拉权重（170.8 GiB / 145 个文件，建议放数据盘）
modelscope download --model hygon/Qwen3.8-Flash-Next-Channel-INT8-w8a8 --local_dir /data2/qwen38-int8-w8a8
```

> 其它方式：也可以用它自带的 Python SDK（`snapshot_download`），或在模型页面上手动取文件——具体以 ModelScope 文档为准。
> **`git clone` 走不通**（本项目实测该仓库的 git 协议端点返回 421），别在那上面浪费时间。

> **下完必须逐文件校验**（本项目踩过"尺寸对但内容坏"的坑，见《优化历程》第 1 节）：
>
> ```bash
> python3 tools/verify_model_download.py --dir /data2/qwen38-int8-w8a8            # 大小 + sha256
> python3 tools/verify_model_download.py --dir /data2/qwen38-int8-w8a8 --size-only  # 只想快速比大小
> ```
>
> 目标机不能访问 modelscope.cn 时就两步走：在能上网的机器上 `--save-manifest ms_files.json --dump-only` 导出清单，拷过去后用 `--manifest ms_files.json` 离线校验。

### 3. 放置本项目文件

```bash
sudo mkdir -p /data2/patches /data2/moe_configs
sudo cp patches/*.py            /data2/patches/
sudo cp configs/config_pleoff.json /data2/patches/       # 可选：只有 serve_cg.sh 的 PLE_OFF 模式才用
sudo cp "configs/E=512,N=160,device_name=K100_AI.json" /data2/moe_configs/
sudo cp scripts/serve_cg_oai.sh /data2/
sudo cp tools/*                 /data2/                  # 验证与压测脚本
sudo chmod +x /data2/*.sh
```

### 4. 起服

```bash
bash /data2/serve_cg_oai.sh
```

脚本会自动：停掉同名旧容器 → 起 4 卡 TP4 容器（挂载上表补丁）→ 等就绪 → 打印 `/v1/models` 与 KV 池信息。

**不用改文件**——本机不一样的地方用环境变量覆盖即可，例如：

```bash
GPUS=0,1,2,3 MODEL_DIR=/data/models/qwen38-int8-w8a8 PORT=8000 bash /data2/serve_cg_oai.sh
```

### 5. 验证

```bash
bash /data2/check_quality.sh            # 输出质量验收（补全 + chat 各若干条）
bash /data2/bench_single_stream.sh      # 单流 TTFT / 解码速率
bash /data2/check_prefix_cache.sh       # 前缀缓存命中与 TTFT 改善（可选）
bash /data2/check_openai_api.sh         # 工具调用 / 思维链分离 / 长上下文
python3 /data2/test_concurrent_a2.py 8388 400 1,4,8,20   # 1/4/8/20 并发：聚合吞吐 / 单流 / TTFT / KV / MTP 接受率
```

### 起服开关（`serve_cg_oai.sh` 顶部环境变量）

所有开关都有合理默认值，**不设也能直接跑**；要按本机情况覆盖：

| 变量 | 默认 | 作用 |
|---|---|---|
| `DOCKER_SOCK` | `unix:///var/run/docker.sock` | docker socket 位置 |
| `IMG` | 官方镜像 | 换成自己 `docker build` 出来的 patched 镜像 |
| `DATA_DIR` | `/data2` | 补丁/配置/编译缓存所在根目录 |
| `MODEL_DIR` | `$DATA_DIR/qwen38-int8-w8a8` | 权重目录 |
| `GPUS` | `4,5,6,7` | 用哪几张卡 |
| `PORT` | `8388` | 服务端口 |

起服**行为**开关：

| 变量 | 默认 | 作用 |
|---|---|---|
| `MAXLEN` | `262144` | 上下文长度 |
| `NSPEC` | `1` | MTP 投机解码步数 |
| `CG_MODE` | `FULL_AND_PIECEWISE` | cudagraph 模式（可 `PIECEWISE` / `NONE`） |
| `TOOL` | `qwen3_xml` | OpenAI 工具调用解析器（`off` 关闭） |
| `REASON` | `qwen3` | 思维链解析器（`off` 关闭） |
| `PC` | `1` | 前缀缓存（`0` 关闭） |

**起服前建议先空跑自检**（只检查环境，不碰任何容器）：

```bash
PREFLIGHT_ONLY=1 bash /data2/serve_cg_oai.sh
```

它会逐条检查 docker socket、镜像是否在本地、权重是否完整、8 个补丁是否齐全、MoE 配置与 `/opt/hyhal`，任何一条不满足就直接停在起服之前。

## 五、目录说明

```
Dockerfile 可选：把补丁烤进官方镜像，构建出"已打补丁"的镜像
patches/   生产补丁（起服时 bind-mount 进容器，覆盖镜像内同名文件）
configs/   MoE 调优配置 + 关闭 N-gram 层的 config 变体
scripts/   serve_cg_oai.sh（推荐）/ serve_cg_mtp.sh / serve_cg.sh（更保守的基线）
tools/     验证与压测：质量、单流速度、并发 A 测、内核单测、复现脚本
diffs/     补丁相对官方镜像原文件的 unified diff（便于审阅改动）
优化历程与实测.md   完整踩坑记录 + 每阶段实测性能 + 对输出质量的影响
```

`tools/` 明细：

| 脚本 | 用途 |
|---|---|
| `check_quality.sh` | 输出质量验收（6 条补全 + 2 条 chat，贪心解码） |
| `bench_single_stream.sh` | 单流 TTFT / 解码速率（短问答 + 长生成） |
| `bench_prefill.sh` | 单请求 prefill 速率（不同 prompt 长度） |
| `test_concurrent_a2.py` | **1/4/8/20 并发 A 测**：聚合吞吐、单流速率、TTFT、KV 峰值、MTP 接受率 |
| `check_openai_api.sh` | OpenAI 兼容面：`max_model_len`、工具调用、思维链分离、长 prompt |
| `check_openai_api_full.sh` | 同上加强版：原始 SSE 字段、流式工具调用、`content` 是否干净 |
| `check_prefix_cache.sh` | 前缀缓存命中率与 TTFT 改善（同 prompt 连问） |
| `check_prefix_cache_multiturn.sh` | 多轮历史逐轮加长形态下的复用率 |
| `verify_model_download.py` | **按 ModelScope 官方 sha256 逐文件校验权重**（支持离线：`--save-manifest` / `--manifest`） |
| `validate_align.py` / `validate_router.py` / `validate_rms.py` / `validate_ple_int8.py` | 各补丁的数值对拍（对齐填充、路由融合、rms_norm、PLE int8） |
| `cg_capture_test.py` | 秒级 CUDA graph 捕获冒烟测试（替代分钟级起服验证） |
| `kernel_battery.py` / `silu_fix_probe.py` / `repro_moe*.py` | 失效内核的定位与复现脚本 |

## 六、已知限制

- 起服后**第一个请求是冷启动**（含新图编译/缓存预热），速率与延迟都不代表稳态，**测速必须跑第二遍**。
- 改动 `max_model_len` 或前缀缓存开关会改变 torch.compile 的 key，触发**全量重编译（4 卡 TP4 约 27 分钟）**。改完请留足时间。
- 极短提示词（1～2 token）在贪心解码下输出不完全确定，属于这套栈的既有现象。
- 前缀缓存在"同前缀大量并发"场景收益显著，但在"多轮对话逐轮加长历史"场景只复用约 40%，具体见《优化历程》第 8 节。

## 七、许可与致谢

### 许可

- `patches/` 目录：**Apache-2.0**（vLLM 源码修改版，见 `patches/NOTICE.md`）
- 仓库其余内容：**0BSD**（见 `LICENSE`）
- 模型权重与容器镜像不随本仓库分发，版权与许可归各自发布方

### 致谢

感谢 **Kimi K3**、**DeepSeek V4.1-Flash** 在调试过程中提供的协助，以及 **序言工作台** 提出的 Agent 长上下文场景需求——这套优化就是围绕它的真实负载磨出来的。
