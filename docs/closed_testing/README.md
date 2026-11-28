# Vexel 0.6.0b1 内测说明（给测试者）

感谢参加 Vexel 的小范围内测。Vexel 是一个分析**微流控液滴录像**的软件：数液滴、测粒径和体积、数每个液滴里的细胞（包裹率与泊松统计）、判读荧光阴阳性（数字 PCR 等），可以分析录好的视频，也可以边采集边分析。每个结果都附有复核图和完整的溯源记录。

这是**内测版**：功能已经齐全，但只在很少的真实数据上验证过。我们最需要的是**你实验室的真实录像**跑出来的结果，以及它哪里错了、哪里让你困惑。

## 内测安排

| 项 | 内容 |
| :--- | :--- |
| 时间 | 约 2 周，期间约 2–4 小时 |
| 你需要 | 一台 macOS（Apple Silicon 或 Intel）、Windows 10/11 或 Linux 电脑；几段自己的液滴录像 |
| 你会做 | 安装 → 用示例数据熟悉 → 分析自己的数据 → 反馈 |
| 联系 | 组织者会告诉你用哪个微信群/邮箱/GitHub 提交反馈 |

## 适用范围

**适合**：液滴（或气泡、珠子）在直通道里**流过**视野的明场录像，液滴大致是圆的；可以带一个荧光通道；可以是视频（.avi/.mp4/.mov）、TIFF/OME-TIFF 堆栈或图像序列文件夹。

**目前不适合**（遇到请告诉我们，记为需求）：液滴不动的静态阵列或腔室芯片、长条形塞子或严重变形的液滴、双重乳液、多色荧光判读、3D 数据、各通道分存为不同文件。

## 1. 安装（约 10–20 分钟）

你会收到三个文件，放在同一个文件夹里：

| 文件 | 作用 |
| :--- | :--- |
| `vexel-0.6.0b1-py3-none-any.whl` | Vexel 本身 |
| `vexel-0.6.0b1-constraints.txt` | 我们测试过的全部依赖的确切版本，安装时用它，保证你装到的和我们测的是同一套 |
| `SHA256SUMS.txt` | 前两个文件的校验值 |

先在这个文件夹里打开终端（Windows 用 PowerShell），核对文件完整（可选，但推荐）：

- macOS / Linux：`shasum -a 256 -c SHA256SUMS.txt`（每行应显示 OK）
- Windows：`Get-FileHash .\vexel-0.6.0b1-py3-none-any.whl`，与 `SHA256SUMS.txt` 里的值对比

然后按你所在的网络环境选一种安装方式。两种方式装出来的软件完全相同。

### 1A. 海外网络

1. 安装 [uv](https://docs.astral.sh/uv/)（Python 工具管理器，会自动准备 Python 3.12）：
   - macOS / Linux：`curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows：`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. **关闭并重新打开**终端，回到放安装文件的文件夹，安装并启动：

```bash
uv tool install ./vexel-0.6.0b1-py3-none-any.whl -c ./vexel-0.6.0b1-constraints.txt
vexel gui
```

### 1B. 中国大陆网络

uv、Python 和依赖包默认都从境外服务器下载，在大陆可能很慢或失败。下面改用中国科学技术大学（USTC）和清华大学的镜像。

**macOS / Linux**（在“终端”里依次运行）：

```bash
# 1. 从 USTC 镜像安装 uv
export UV_DOWNLOAD_URL=https://mirrors.ustc.edu.cn/github-release/astral-sh/uv/LatestRelease
curl -LsSf https://mirrors.ustc.edu.cn/github-release/astral-sh/uv/LatestRelease/uv-installer.sh | sh

# 2. 让 uv 以后都从国内镜像下载 Python 和依赖包（写入启动配置，只需做一次）
echo 'export UV_PYTHON_INSTALL_MIRROR=https://mirrors.ustc.edu.cn/github-release/astral-sh/python-build-standalone' >> ~/.zshrc
echo 'export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple' >> ~/.zshrc
```

（Linux 或使用 bash 时把 `~/.zshrc` 换成 `~/.bashrc`。）**关闭并重新打开**终端，回到放安装文件的文件夹：

```bash
uv tool install ./vexel-0.6.0b1-py3-none-any.whl -c ./vexel-0.6.0b1-constraints.txt
vexel gui
```

**Windows**：

1. 下载 uv：<https://mirrors.ustc.edu.cn/github-release/astral-sh/uv/LatestRelease/uv-x86_64-pc-windows-msvc.zip>（ARM 电脑下载 `uv-aarch64-pc-windows-msvc.zip`），解压后把 `uv.exe` 放到一个固定位置，例如 `C:\Users\你的用户名\.local\bin`。
2. 在 PowerShell 里运行（只需一次，设置后**重新打开** PowerShell）：

```powershell
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:USERPROFILE\.local\bin", "User")
[Environment]::SetEnvironmentVariable("UV_PYTHON_INSTALL_MIRROR", "https://mirrors.ustc.edu.cn/github-release/astral-sh/python-build-standalone", "User")
[Environment]::SetEnvironmentVariable("UV_DEFAULT_INDEX", "https://pypi.tuna.tsinghua.edu.cn/simple", "User")
```

3. 回到放安装文件的文件夹：

```powershell
uv tool install .\vexel-0.6.0b1-py3-none-any.whl -c .\vexel-0.6.0b1-constraints.txt
uv tool update-shell
vexel gui
```

**镜像不可用时**：清华源可以换成阿里云 `https://mirrors.aliyun.com/pypi/simple`；Python 镜像可以换成 `https://registry.npmmirror.com/-/binary/python-build-standalone`。USTC 只保留 Python 的最新一批构建，个别文件不在镜像上时会自动转去 GitHub 下载，会慢一些但能完成。

### 被邀请访问代码仓库时（两地都适用）

```bash
git clone https://github.com/Chris-zheng/vexel.git
cd vexel
uv sync
uv run vexel gui
```

用这种方式时，下文的 `vexel ...` 都写成 `uv run vexel ...`。大陆网络请先按 1B 设置好镜像。

### 常见问题

- **`vexel` 命令找不到**：运行 `uv tool update-shell`，然后重新打开终端。
- 第一次安装要下载约 300 MB，可能要几分钟。打不开时，请把终端里的完整报错截图发给我们。

## 2. 熟悉（约 15 分钟，请计时）

> 这一步也是我们的可用性测试：**请先不看手册**，按界面上方引导栏的提示走，记下用了多少分钟、在哪里卡住。

1. 启动后先出现模块选择窗口，点“微流控液滴”的 **打开**（勾选“下次启动时直接打开上次使用的模块”后以后会跳过这一步）。“SEM 扫描电镜”模块还在开发中。
2. 点引导栏里的 **试用示例数据**（或 帮助 → 打开示例数据）。
3. 试着得到：液滴的平均直径、每个液滴平均有几个细胞。
4. 打开 HTML 报告看一看。

## 3. 分析你自己的数据（主要任务）

请尽量用 **3 段以上**不同条件的录像（不同液滴大小、流速、有无细胞、有无荧光）。对每一段：

1. **载入**：把文件拖进窗口，或点 **选择文件…**（图像序列点 **图像序列文件夹…**）。**填上像素尺寸（µm/像素）和帧间隔**——不填的话结果以像素和帧为单位。视频文件里的帧率常常不对（比如慢放），请以相机设置为准。
2. **设置**：在“想测什么？”里用一句话写目标（如“数每个液滴里的细胞”“ddPCR 阳性比例”），点 **自动规划…**，看它的理由是否合理，再 **采用**；也可以手动勾选。
3. 点 **快速预览**，看叠加图上液滴圈得对不对。不对时试试 **高级设置**，或 画笔训练里的 **训练…**（在图上涂几笔“液滴”和“背景”）。
4. 点 **开始分析** 处理全部帧，看结果卡片，点 **打开报告**。
5. **复核**：看“可疑液滴”下列出的问题，点 **复核与纠错…** 逐个确认；有错就改，改完点 **应用并重算**。
6. 把结果和你**已知的答案**比一比（人工数过的液滴数、显微镜量过的粒径、已知的包裹率或阳性率），写进反馈表。

**如果你有同一实验不同条件的多段录像**（推荐）：在主窗口用其中一段把设置调好，点菜单 **文件 → 批量分析（多个文件、分组统计）…**，选择文件夹，检查“组”（条件）和“重复”（独立芯片/批次/日期），开始分析，然后看批量报告里的分组比较是否符合你的预期。

完整说明见 [用户手册](../user_manual.md)，尤其是第 2 节（界面，批量见 2.10）和第 7 节（调参）。

## 4. 反馈

每遇到一个问题，或每分析完一段数据，填一份 [反馈表](feedback_template.md)。如果是**出错、崩溃或结果明显不对**，请附上诊断包：

- 界面里：帮助 → **导出诊断包…**（包含当前这次分析）
- 或命令行：`vexel bugreport <结果文件夹> -o 诊断包.zip --note "发生了什么"`

诊断包里有软件版本、操作系统、设置、配方、汇总结果和可疑液滴列表；**不含图像、视频、报告和任何密钥**，文件路径只保留文件名。发之前可以解压看一看。

**愿意的话，请把出问题的那段录像也发给我们**（或其中 10–30 秒的片段）。这对改进最有帮助。我们只把它用于内部开发和测试，不会公开，不会转交他人；你可以随时要求删除。

内测结束时还有一份简短的问卷（见反馈表末尾）。

## 5. 隐私与数据

- 默认情况下，**所有分析都在你的电脑上完成**，不联网，不上传任何东西。
- **自动规划**在 0.8 测试版里只用离线规则引擎，不联网（云端模型以后经 Vexel 网关提供）。旧版本中：如果你在 帮助 → 规划模型设置 里选了云端大模型（Claude、DeepSeek、通义千问、Gemini），Vexel 会把数据“指纹”里的**数字**（尺寸、亮度统计、流速等）和你写的目标发给该服务商，**不发图像、不发文件路径**；需要你自己的 API 密钥，费用由你的账户承担。不想有任何网络请求：保持“离线规则引擎”，或设置环境变量 `VEXEL_OFFLINE=1`。
- Vexel 不收集使用统计，不会“打电话回家”。我们只能看到你主动发来的反馈、诊断包和录像。

## 6. 已知问题

见 [已知问题](known_issues.md)。遇到列在其中的问题不必再报，但如果它影响了你的实验，请告诉我们影响有多大。

## 7. 卸载

- 用安装文件装的：`uv tool uninstall vexel`
- 用代码仓库装的：删除 `vexel` 文件夹
- 两种方式都可再删除 `~/.vexel`（配方记忆与模型设置）
