# Unit00 Hermes Agent 常用指令詳細教學

## 課程目標

本單元是 [Unit00 Hermes Agent 安裝與使用教學](Unit00_Hermes_Agent_Setup.md) 的續篇。前一份文件帶領同學完成安裝與初次設定，本文件則聚焦於 **常用指令的深入說明與實戰範例**——每一個指令都會說明其用途、語法、常用參數、執行後的預期輸出，並搭配化工學習情境的完整範例。

### 學習目標

- 熟練 Hermes Agent 的對話管理指令（`hermes`、`hermes --continue`、`hermes -r`）
- 掌握設定管理指令（`hermes config`、`hermes model`、`hermes tools`）
- 學會技能系統操作（`hermes skills`）與訊息閘道設定（`hermes gateway`）
- 能使用診斷與維護指令（`hermes doctor`、`hermes update`、`hermes pm`）自行排除問題
- 熟悉對話內斜線指令（Slash Commands）的日常應用
- 透過完整實戰範例，將指令應用於化工資料分析工作流程

### 前置需求

- 已依照 [Unit00 Hermes Agent 安裝與使用教學](Unit00_Hermes_Agent_Setup.md) 完成 Hermes Agent 安裝
- 已完成至少一次模型供應商設定，並成功進行一次對話
- 作業系統：Windows 10 / 11（本文件以 PowerShell 為主）

### 指令總覽地圖

```mermaid
flowchart TD
    A[Hermes 常用指令] --> B[對話管理]
    A --> C[設定管理]
    A --> D[功能擴充]
    A --> E[維護診斷]
    B --> B1["hermes<br/>hermes --tui<br/>hermes --continue<br/>hermes -r"]
    C --> C1["hermes model<br/>hermes config<br/>hermes tools"]
    D --> D1["hermes skills<br/>hermes gateway<br/>hermes pm"]
    E --> E1["hermes doctor<br/>hermes update<br/>hermes sessions<br/>hermes uninstall"]
```

---

## 1. 對話管理指令

### 1.1 `hermes`：開始新對話

**用途**：啟動一次全新的代理對話，這是最基本的指令。

**語法**：

```powershell
hermes [選項] [子指令]
```

**常用參數**：

| 參數 | 說明 |
|------|------|
| （無參數） | 進入互動模式，直接輸入提示開始對話 |
| `--tui` | 以現代終端介面（TUI）啟動，介面較友善，適合偏好終端機的使用者 |
| `--cli` | 強制使用傳統 REPL 介面 |
| `--yolo` | 略過危險指令核准提示（**危險，慎用**） |
| `--in <路徑>` | 先切換到指定目錄再啟動或接續對話（作為代理的工作目錄） |
| `-m, --model <模型>` | 暫時指定本次使用的模型 |
| `-z, --oneshot <提示>` | 單次模式：送出一個提示，只輸出最終回覆後結束 |
| `-w, --worktree` | 在隔離的 git worktree 中啟動 |

> **注意**：主指令 `hermes` 的工作目錄旗標是 `--in`；`--cwd` 僅用於 `hermes desktop`（見 3.4）。實際可用旗標請以 `hermes --help` 為準，不同版本可能略有差異。

**範例 1：互動模式開始對話**

```powershell
hermes
```

執行後會進入對話提示符，直接輸入問題即可：

```text
> 你好，請自我介紹，並列出你目前可用的工具。
```

**範例 2：單次任務（回答後即結束）**

```powershell
hermes -z "用一句話說明什麼是 CSTR"
hermes chat -q "檢查 D:\MyGit\ChemE-3590 資料夾結構，列出所有 .ipynb 檔案" --oneshot
```

> **說明**：`hermes chat -q "提示"` 在一般終端機中只是把提示當作第一則訊息，之後仍會進入互動對話；要「回答後直接結束」需加上 `--oneshot`（或 `-Q`），`-z` 則等同於單次模式。若想讓提示內容完全不受 PowerShell 引號或特殊字元影響，可改用 `--query-file <檔案>`。

**範例 3：指定工作目錄與 TUI 介面**

```powershell
hermes --tui --in D:\MyGit\ChemE-3590\Part_1
```

> **化工情境**：做反應動力學作業時，用 `--in` 直接把工作目錄指到作業資料夾，代理讀寫檔案就不用每次都寫完整路徑。

### 1.2 `hermes --continue`：接續最近對話

**用途**：接續最近一次的對話，或依名稱接續指定對話，保留完整上下文。

**語法**：

```powershell
hermes --continue [對話名稱]
hermes -c [對話名稱]
```

**範例**：

```powershell
hermes -c                      # 接續最近一次對話
hermes -c "kinetics-homework"  # 依名稱接續（取該名稱最新的一筆）
```

**使用時機**：昨天做到一半的資料清理任務，今天開終端機直接 `hermes -c`，代理還記得昨天讀過哪些檔案、處理到哪一步。

### 1.3 `hermes -r`：依標題或 ID 接續對話

**用途**：以 `--resume`（簡寫 `-r`）依「對話 ID 或標題」接續指定對話，適合管理多個並行專案。

**語法**：

```powershell
hermes -r "<對話標題>"
hermes -r <session_id>
hermes -r latest          # 等同 hermes -c
```

**範例**：

```powershell
hermes -r "kinetics-homework"
```

> **搭配技巧**：先用 `/title kinetics-homework` 為對話命名，日後才能用 `-r` 精準找回。標題建議使用英文小寫加連字號，方便輸入與搜尋。對話 ID 可由 `hermes sessions list` 取得。

### 1.4 `hermes sessions`：對話清單管理

**用途**：列出與管理所有歷史對話。

**常用子指令**：

| 子指令 | 說明 |
|--------|------|
| `hermes sessions list` | 列出最近的對話 |
| `hermes sessions browse` | 互動式選擇器：瀏覽、搜尋並接續對話 |
| `hermes sessions rename <ID> <標題>` | 設定或修改對話標題 |
| `hermes sessions export` | 匯出對話為 JSONL、Markdown 等格式 |
| `hermes sessions delete <ID>` | 刪除指定對話（不可復原，請謹慎） |

**範例**：

```powershell
hermes sessions list
```

**預期輸出（示意，欄位與格式依版本而異；ID 為一長串識別碼，可複製給 `-r` 使用）**：

```text
Title                 Workspace        Last Active   ID
─────────────────────────────────────────────────────────────────────
kinetics-homework     Part_1           18m ago       local-87992b39dc6c...
reactor-temp-analysis reactor-data     1d ago        local-56e36e6fc63e...
first-chat            Hermes           3d ago        local-eb2ddca66e93...
```

---

## 2. 設定管理指令

### 2.1 `hermes model`：選擇供應商與模型

**用途**：互動式選擇 LLM 供應商（Provider）與模型（Model），或進行 OAuth 登入。

**語法**：

```powershell
hermes model
```

**互動流程**：

1. 列出所有支援的供應商（Custom Endpoint、Nous Portal、Anthropic、OpenAI、OpenRouter 等）
2. 選擇供應商後，若需要認證會引導登入或輸入 API 金鑰
3. 列出該供應商可用的模型清單，選擇一個設為預設

> **本課程**：選 **Custom Endpoint**，填入教師提供的實驗室服務網址（base URL）、課程 API 金鑰與模型名稱（詳見 [Unit00 安裝教學](Unit00_Hermes_Agent_Setup.md) 4.1 與 4.5）。其他供應商僅供課外參考。

**常見情境對照**：

| 情境 | 操作 |
|------|------|
| 第一次設定（本課程） | 執行 `hermes model`，選 Custom Endpoint，填入實驗室端點 |
| 課外想用 Nous Portal | 執行 `hermes setup --portal`（不用於課程作業） |
| 想從 Claude 換成 GPT | 執行 `hermes model`，選 OpenAI 供應商 |
| API 金鑰過期，對話報錯 | 執行 `hermes model` 重新認證 |

> **成本提醒**：切換模型會重置提示快取（prompt cache），長對話中頻繁切換會增加成本。若只是想換個模型試試，建議另開新對話。

### 2.2 `hermes config`：設定值的讀寫

**用途**：直接讀寫個別設定值，適合腳本化或快速調整。

**語法**：

```powershell
hermes config set <key> <value>
hermes config get <key>
hermes config check
hermes config migrate
```

**常用設定項範例**：

```powershell
# 設定預設模型（<provider/model> 請換成 hermes model 清單中的實際名稱）
hermes config set model <provider/model>

# 讓終端機指令在 Docker 容器中執行（沙箱隔離）
hermes config set terminal.backend docker

# 查看目前預設模型
hermes config get model

# 查看設定檔與 .env 的實際位置
hermes config path
hermes config env-path
```

> **不要把 API 金鑰直接打在命令列**：`hermes config set OPENROUTER_API_KEY sk-or-...` 這類寫法會讓金鑰以明文留在 PowerShell 歷史紀錄（`ConsoleHost_history.txt`）。金鑰請改用 `hermes model` 的互動式輸入或桌面 GUI 的 Settings → Providers；若已誤用，請立即至供應商後台更換該金鑰。

**設定檢查與修復**：

```powershell
hermes config check      # 檢查設定是否有缺失或過時
hermes config migrate    # 自動遷移／修復舊版設定
```

> **機密分離原則**：`hermes config set` 會自動把金鑰類的值寫入 `HERMES_HOME\.env`、一般設定寫入 `config.yaml`，不需要（也不應該）手動編輯這些檔案。

### 2.3 `hermes tools`：工具啟用管理

**用途**：檢視與設定代理可使用哪些工具（終端機、檔案、網頁搜尋、瀏覽器等）。

**語法**：

```powershell
hermes tools
```

**使用建議**：

- 初學者建議維持預設，不要一次關閉太多工具
- 若代理誤用工具造成困擾，可在此關閉特定工具
- 想要瀏覽器自動化能力，需先安裝 `agent-browser`（見 3.3 節 `hermes pm`）

> 也可用 `hermes setup tools` 只執行「工具」這一段的設定精靈。其他可單獨執行的區段包含 `model`、`tts`、`terminal`、`gateway`、`telemetry`、`agent`。

### 2.4 `hermes setup`：完整設定精靈

**用途**：互動式完成所有設定（供應商、模型、工具、閘道）。

**三種模式**：

| 模式 | 指令 | 適用對象 |
|------|------|---------|
| Full Setup | `hermes setup` | 逐一設定；供應商選 Custom Endpoint，雲端工具先略過（**本課程建議**） |
| Blank Slate | 精靈中選擇 | 只開最低限度，同樣適合本課程 |
| Quick Setup | `hermes setup --portal` | 有 Nous Portal 帳號（課外參考，本課程不使用） |

> **補充**：已安裝過的環境執行 `hermes setup` 會以現有值為預設重新跑完整精靈；加上 `--quick` 則只詢問缺漏或未設定的項目；`--reset` 會把設定重置為預設值（請謹慎）。

**範例**：

```powershell
hermes setup
```

選擇 **Full Setup**：供應商選 Custom Endpoint 並填入教師提供的實驗室端點；遇到網頁搜尋、影像生成、TTS、雲端瀏覽器等需連到雲端的工具時先略過不啟用（本課程規定見 [Unit00 安裝教學](Unit00_Hermes_Agent_Setup.md) 4.5）。

---

## 3. 功能擴充指令

### 3.1 `hermes skills`：技能系統操作

**用途**：瀏覽、搜尋與安裝技能（Skills）。技能是「隨需載入」的程序性知識文件，教代理如何完成特定任務。

**語法**：

```powershell
hermes skills browse                    # 瀏覽所有可用技能（分頁）
hermes skills search <關鍵字>           # 依關鍵字搜尋技能
hermes skills inspect <來源/路徑>       # 安裝前預覽技能內容
hermes skills install <來源/路徑>       # 安裝技能（安裝前自動安全掃描）
hermes skills list                      # 列出已安裝技能
hermes skills update                    # 更新已安裝的技能
```

**範例 1：搜尋資料分析相關技能**

```powershell
hermes skills search pandas
```

**範例 2：安裝技能**

```powershell
hermes skills install openai/skills/k8s
```

**安裝後的效果**：每個已安裝的技能會自動成為一個斜線指令，例如安裝 `k8s` 技能後：

```text
/k8s deploy the staging manifest
```

> **化工情境**：把「讀取實驗 CSV → 清理離群值 → 繪製趨勢圖 → 輸出統計摘要」的固定流程存成自訂技能，日後一個斜線指令就能重複執行整個分析流程。

### 3.2 `hermes gateway`：訊息閘道設定

**用途**：將 Hermes 連接到通訊平台（Telegram、Discord、Slack、WhatsApp、Signal、Email、Microsoft Teams 等），讓代理變成隨時可用的機器人助理。

> **本課程說明**：訊息閘道會把對話送到第三方通訊平台，屬課外參考；課程作業與共同任務不使用（見 [Unit00 安裝教學](Unit00_Hermes_Agent_Setup.md) 4.5）。

**語法**：

```powershell
hermes gateway setup    # 互動式平台設定
hermes gateway run      # 在前景執行閘道（Windows 建議使用此方式）
hermes gateway status   # 查看閘道連線狀態
```

> **Windows 注意**：`hermes gateway install` / `start` 是安裝並啟動 systemd / launchd 背景服務，僅適用於 Linux / macOS。在 Windows 上請用 `hermes gateway run`，並保持該終端機視窗開啟。

**範例：設定 Telegram 機器人**

```powershell
hermes gateway setup
```

互動流程會引導你：

1. 選擇平台（Telegram）
2. 輸入 BotFather 給予的 Bot Token
3. 設定允許清單（哪些使用者可以跟你的代理對話）
4. 以 `hermes gateway run` 啟動閘道，並用 `hermes gateway status` 驗證連線

**驗證狀態**：

```powershell
hermes gateway status
```

> **安全提醒**：Bot Token 屬於機密，`hermes gateway setup` 會自動將其寫入 `HERMES_HOME\.env`，切勿手動貼到對話中或上傳 Git。

### 3.3 `hermes pm`：套件管理器

**用途**：管理 Hermes 內建的執行環境元件（Python、Node.js、瀏覽器工具等）。

**常用子指令**：

| 子指令 | 說明 |
|--------|------|
| `hermes pm install agent-browser` | 安裝瀏覽器自動化工具（含 Chromium） |
| `hermes pm install cua-driver` | 安裝電腦操作驅動（滑鼠鍵盤控制） |

**範例**：安裝時若用了 `-SkipBrowser` 旗標跳過瀏覽器工具，日後想補裝：

```powershell
hermes pm install agent-browser
```

### 3.4 `hermes desktop`：啟動桌面應用

**用途**：從命令列啟動 Hermes 桌面應用，沿用現有的設定、金鑰、對話與技能。

**語法**：

```powershell
hermes desktop [--cwd <路徑>]
```

**範例**：

```powershell
# 啟動桌面應用並直接開啟專案目錄
hermes desktop --cwd D:\MyGit\ChemE-3590
```

> **多介面共用**：桌面應用、CLI、TUI 共用同一份狀態。你可以在 CLI 開始一個對話，稍後用 `hermes desktop` 打開桌面應用接續，反之亦然。

---

## 4. 維護與診斷指令

### 4.1 `hermes doctor`：萬用診斷

**用途**：全面檢查安裝與設定狀態，明確指出缺少什麼以及如何修正。**遇到任何問題，第一個該執行的就是它。**

**語法**：

```powershell
hermes doctor
```

**預期輸出（示意，實際文字與項目依版本而異）**：

```text
✓ Hermes core installed
✓ Python runtime available
✗ No model provider configured
  → Run: hermes model
✓ Terminal tools available
✗ Browser tools not installed
  → Run: hermes pm install agent-browser
```

**排查順序建議**（與安裝教學 11.2 一致）：

1. `hermes doctor` — 先看整體健康狀態
2. `hermes model` — 供應商或金鑰問題
3. `hermes config check` — 設定值缺失或過時
4. `hermes sessions list` — 找不到舊對話時

> 另可用 `hermes --version` 確認版本與安裝目錄，用 `hermes logs errors` 檢視錯誤日誌。

### 4.2 `hermes update`：更新版本

**用途**：更新 Hermes Agent 到最新版本。

**語法**：

```powershell
hermes update
```

> **更新後注意**：若更新後出現設定異常，依序執行 `hermes config check` 與 `hermes config migrate` 修復。

### 4.3 `hermes uninstall`：解除安裝

**用途**：移除 Hermes Agent，可選擇移除範圍。

**移除範圍對照**：

| 指令 | 移除範圍 |
|------|---------|
| `hermes uninstall --gui` | 僅桌面應用（保留代理、設定與對話） |
| `hermes uninstall` | 應用與代理（保留設定與對話資料） |
| `hermes uninstall --data` | 僅移除使用者資料（保留已安裝的程式） |
| `hermes uninstall --full` | 全部移除（**含所有使用者資料，不可復原**） |

> **先預覽再執行**：任何解除安裝指令都可加上 `--dry-run`，只列出將被移除的內容而不實際變更，例如 `hermes uninstall --full --dry-run`。

> **危險操作提醒**：`--full` 會刪除 `HERMES_HOME` 下所有資料（含對話記錄、記憶、技能、API 金鑰）。執行前請確認已備份重要對話或技能檔案。

---

## 5. 對話內斜線指令（Slash Commands）

在對話輸入框中輸入 `/` 會出現自動完成選單。這些指令在對話「過程中」使用，與前面的終端機指令不同。

### 5.1 常用斜線指令詳解

| 指令 | 說明 | 使用時機 |
|------|------|---------|
| `/help` | 顯示所有可用指令 | 忘記指令時 |
| `/tools` | 列出目前可用的工具 | 確認代理有哪些能力 |
| `/model` | 互動式切換模型 | 對話中途想換模型 |
| `/save` | 儲存目前對話 | 完成重要階段時 |
| `/title <名稱>` | 為對話命名 | 開始長任務前（方便日後 `-r` 接續） |
| `/compress` | 壓縮（摘要）對話歷史 | 對話很長、token 用量高時 |
| `/usage` | 查看 token 用量 | 監控成本 |
| `/verbose` | 切換工具輸出顯示模式 | 想看（或隱藏）工具執行細節 |
| `/yolo` | 切換 YOLO 模式 | **危險**：略過危險指令核准，僅在受控環境使用 |
| `/approvals [manual\|smart\|off]` | 檢視或切換核准模式 | 想調整危險指令的詢問程度時 |

**範例：長對話的成本管理流程**

```text
> /title reactor-analysis        ← 命名對話
> /usage                         ← 檢查用量
> （進行多輪資料分析...）
> /compress                      ← token 變多時壓縮歷史
> /usage                         ← 確認壓縮後用量下降
```

### 5.2 斜線指令 vs 終端機指令

| 面向 | 斜線指令（`/model`） | 終端機指令（`hermes model`） |
|------|---------------------|------------------------------|
| 執行位置 | 對話輸入框內 | PowerShell 終端機 |
| 影響範圍 | 通常僅目前對話 | 全域設定（寫入 config） |
| 適用場景 | 對話進行中的即時調整 | 對話開始前的環境準備 |

> **記憶口訣**：斜線指令管「這場對話」，終端機指令管「整個代理」。

---

## 6. 實戰範例：化工資料分析工作流程

以下用一個完整情境串連所有指令：**分析反應器溫度數據並產生報告**。

### 6.1 情境描述

你有一批反應器溫度感測器數據（`reactor_temp.csv`），需要：清理離群值、繪製趨勢圖、計算統計摘要，並在隔天繼續追加分析。

### 6.2 步驟一：環境準備（終端機）

```powershell
# 1. 確認環境健康
hermes doctor

# 2. 以專案目錄為工作目錄，開啟 TUI 對話
hermes --tui --in D:\ChemE-projects\reactor-data
```

### 6.3 步驟二：對話內操作

```text
> /title reactor-temp-analysis
> 讀取 reactor_temp.csv，檢查缺失值與離群值，
  繪製溫度對時間的趨勢圖並存成 PNG，輸出統計摘要。
```

代理會自動：讀檔 → 用 `execute_code` 執行 Python（pandas / matplotlib）→ 產生圖檔與摘要。

> **多行輸入提示**：在 TUI 中輸入多行提示時，以 `Alt+Enter` 或 `Ctrl+J` 換行，最後才按 `Enter` 送出。

### 6.4 步驟三：隔天接續（終端機）

```powershell
# 方式一：接續最近對話
hermes -c

# 方式二：依標題精準接續
hermes -r "reactor-temp-analysis"
```

接續後直接追加新需求：

```text
> 把離群值判定改用 3 倍標準差，重新繪圖並比較前後差異。
```

### 6.5 步驟四：流程固化為技能（進階）

若此分析流程會反覆執行，可請代理把做法存成技能：

```text
> 把剛剛的溫度數據分析流程存成一個技能，命名為 temp-analysis
```

日後在新資料上只需：

```text
> /temp-analysis 分析 batch_2026_10.csv
```

### 6.6 完整指令時序圖

```mermaid
sequenceDiagram
    participant U as 使用者
    participant T as 終端機
    participant H as Hermes Agent
    U->>T: hermes doctor
    T-->>U: 環境檢查結果
    U->>T: hermes --tui --in 專案路徑
    T->>H: 啟動對話
    U->>H: /title reactor-temp-analysis
    U->>H: 分析 reactor_temp.csv
    H->>H: 讀檔 + execute_code + 繪圖
    H-->>U: 趨勢圖 PNG + 統計摘要
    Note over U,H: 隔天
    U->>T: hermes -r "reactor-temp-analysis"
    T->>H: 接續對話（含完整上下文）
    U->>H: 改用 3 倍標準差重新分析
    H-->>U: 新圖表 + 前後比較
```

---

## 7. 常用指令速查表

### 7.1 終端機指令（依使用頻率排序）

| 頻率 | 指令 | 一句話說明 |
|------|------|-----------|
| ★★★ | `hermes` | 開始對話 |
| ★★★ | `hermes --tui` | 以 TUI 介面開始對話（偏好終端機者推薦） |
| ★★★ | `hermes -c` | 接續最近對話 |
| ★★★ | `hermes -r "<標題或ID>"` | 依標題或 ID 接續對話 |
| ★★☆ | `hermes model` | 選擇供應商與模型 |
| ★★☆ | `hermes doctor` | 診斷問題（萬用） |
| ★☆☆ | `hermes setup --portal` | Nous Portal 快速設定（課外參考） |
| ★★☆ | `hermes config set/get` | 讀寫設定值 |
| ★★☆ | `hermes desktop` | 啟動桌面應用（可加 `--cwd <路徑>`） |
| ★☆☆ | `hermes sessions list` | 列出對話清單 |
| ★☆☆ | `hermes sessions rename <ID> <標題>` | 為對話命名 |
| ★☆☆ | `hermes skills browse/search/install` | 技能管理 |
| ★☆☆ | `hermes gateway setup/run/status` | 訊息閘道 |
| ★☆☆ | `hermes tools` | 工具啟用管理 |
| ★☆☆ | `hermes pm install <套件>` | 安裝環境元件 |
| ★☆☆ | `hermes update` | 更新版本 |
| ★☆☆ | `hermes config check/migrate` | 設定檢查與修復 |
| 慎用 | `hermes --yolo` / `/yolo` | 略過核准提示（危險） |
| 慎用 | `hermes uninstall [--full]` | 解除安裝（`--full` 不可復原） |

### 7.2 對話內斜線指令

| 指令 | 一句話說明 |
|------|-----------|
| `/help` | 顯示所有指令 |
| `/title <名稱>` | 命名對話（方便 `-r` 接續） |
| `/compress` | 壓縮對話歷史省 token |
| `/usage` | 查看 token 用量 |
| `/model` | 對話中途切換模型 |
| `/save` | 儲存對話 |
| `/tools` | 列出可用工具 |
| `/verbose` | 切換工具輸出顯示 |

---

## 8. 常見錯誤與排除

| 症狀 | 可能原因 | 解決指令 |
|------|---------|---------|
| PowerShell 顯示「無法辨識 'hermes' 詞彙…」（英文系統為 `The term 'hermes' is not recognized...`） | 安裝後尚未重新載入 PATH | **重開** PowerShell；仍無效則檢查使用者 PATH 是否含 `%LOCALAPPDATA%\hermes\bin`，或重新執行安裝程式 |
| `API key not set` | 金鑰未設定或過期 | `hermes model` 重新認證 |
| 對話回覆空白 | 供應商認證或模型錯誤 | `hermes model` → `hermes doctor` |
| `hermes -r` 找不到對話 | 對話未命名或設定檔不同 | `hermes sessions list` 確認標題或 ID |
| 更新後設定異常 | 設定格式過時 | `hermes config check` → `hermes config migrate` |
| token 用量暴增 | 對話過長或頻繁切換模型 | `/compress`；必要時另開新對話 |

---

## 9. 小結

本文件深入說明了 Hermes Agent 的常用指令，涵蓋四大類：

1. **對話管理**：`hermes`、`--continue`、`-r`、`sessions` — 日常使用的主戰場
2. **設定管理**：`model`、`config`、`tools`、`setup` — 環境準備與調整
3. **功能擴充**：`skills`、`gateway`、`pm`、`desktop` — 進階能力
4. **維護診斷**：`doctor`、`update`、`uninstall` — 排除問題

**學習建議**：先熟練 ★★★ 等級的四個對話指令，搭配 `/title` 與 `/compress` 兩個斜線指令，就足以應付日常的化工資料分析工作。其餘指令在需要時再回來查閱即可。

> **延伸閱讀**：
> - 安裝與介面導覽：[Unit00 Hermes Agent 安裝與使用教學](Unit00_Hermes_Agent_Setup.md)
> - 化工系技能與外掛推薦：[Unit00 Hermes Agent 技能（Skills）與外掛（Plugins）推薦指南](Unit00_Hermes_Agent_Skills&Plugins.md)
> - 完整官方文件：[Hermes Agent Docs](https://hermes-agent.nousresearch.com/docs)

---

**課程資訊**
- 課程名稱：AI在化工上之應用 (ChemE 3590)
- 課程單元：Hermes Agent 常用指令詳細教學
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-11-05

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

