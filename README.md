# Source Grounding Auditor MVP

这个项目实现了一个“来源支撑审计工具”的可运行 MVP。它不判断观点对错，不输出单一可信度分数，而是把输入文本拆成 atomic claims，分析每条 claim 的来源支撑状态，并输出比例。

## 当前能力

- 输入 AI 回答、文章正文或带 citation 的文本。
- 默认使用 citation-only mode：只分析带 citation 的句子或段落；未带 citation 的内容不进入 claims，也不进入 ratios。
- 抽取并粗略拆分 atomic claims。
- 解析 URL、Markdown citation、脚注和参考文献 URL。
- 将 claim 分为 factual、attribution、judgment、non_claim。
- 将数据作为标签：has_quantitative_data。
- 支持 provided_sources，用于把 citation URL 映射到用户提供的 source text。
- 默认抓取显式 URL。
- 默认搜索无 URL 的 `[1] Reuters ...` 来源说明，发现候选公开来源。
- 输出以下比例：
  - 内容构成比例
  - 来源链终点比例
  - 支撑关系比例
  - 公开事实支撑率
  - 宽松事实支撑率
  - 观点包装率
  - 来源不透明率
  - 引用错配率
- 提供高风险 claim 列表。
- 提供简单 Web UI。

## LLM-first structured classification

本项目使用 LLM first structured classification。启发式规则只用于 citation parsing、schema validation、fallback 和测试，不用于核心语义判断。

`risk_flag` 是底层诊断，不等于 problematic citation。`problematic_citations` 只表示 cited claims 中作者真实主张的、重要的、且证据关系存在实质问题的 citation。

`audit_limited_citations` 只表示本轮无法完成 source support check，不表示 claim 错误。`attribution_supported_citations` 表示 source 支持“某来源说过这件事”，不表示被转述内容本身已被一手事实证明。

所有比例默认都基于 cited claims，响应中的 `summary.ratios_basis` 会写明 `based only on cited claims`。预留字段 `uncited_claim_analysis_enabled` 当前默认为 `false`。

## 重要限制

- 当前 claim extraction 默认使用 Codex CLI 的快速结构化模式，也支持 OpenAI API。
- 当前 source support check 依赖结构化 LLM 判断，不是严格事实核查。
- 默认抓取外部网页。可在 API 请求中传 `enable_url_fetch=false` 关闭显式 URL 抓取。
- 默认搜索无 URL 的来源说明。可在 API 请求中传 `enable_web_search=false` 关闭搜索。
- 当前搜索 provider 使用 no-key DuckDuckGo HTML 搜索，搜索结果质量会影响 discovered source 的准确性。
- 追踪上游来源时，只承认 source 文本中显式出现的 URL，不根据语义相似度猜测 source edge。
- Codex/ChatGPT 订阅不能直接当作 OpenAI API key 使用；如需走订阅通道，本项目通过本机 `codex exec` 接入。

## 安装与运行

```bash
cd source_grounding_auditor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=backend uvicorn app.main:app --reload --app-dir backend
```

打开：

```text
http://127.0.0.1:8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

## LLM claim extraction 测试

默认请求走 Codex CLI 的快速结构化模式。先确认本机 Codex 已登录：

```bash
codex login status
```

然后请求：

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "claim_extraction_mode": "codex",
    "input_text": "OpenAI released GPT-4o in 2024, and it supports text, audio, and image inputs."
  }'
```

默认 Codex 模型是 `gpt-5.3-codex-spark`，可以用环境变量覆盖：

```bash
export CODEX_MODEL="gpt-5.3-codex-spark"
export CODEX_SERVICE_TIER="fast"  # 可选，默认 fast
export CODEX_REASONING_EFFORT="low"  # 可选，默认 low
export CODEX_TIMEOUT_SECONDS="90"  # 可选，默认 90 秒
```

如需测试 OpenAI API 抽取，先在启动后端的同一个 shell 设置 API key：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_MODEL="gpt-4o-mini"  # 可选
PYTHONPATH=backend uvicorn app.main:app --reload --app-dir backend
```

然后请求：

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "claim_extraction_mode": "openai",
    "input_text": "OpenAI released GPT-4o in 2024, and it supports text, audio, and image inputs."
  }'
```

也可以传 `"claim_extraction_mode": "auto"`：优先使用 `OPENAI_API_KEY`，其次使用已登录的 Codex CLI。没有可用 LLM 时会返回配置错误。

## API 示例

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "input_text": "The company reported revenue of $10 billion in its 2024 annual report [source](https://example.com/ar).",
    "provided_sources": [
      {
        "url": "https://example.com/ar",
        "title": "2024 annual report",
        "source_type": "primary_fact_source",
        "access_status": "accessible",
        "extracted_text": "The company reported revenue of $10 billion in its 2024 annual report."
      }
    ]
  }'
```

如果输入只有编号来源说明、没有 URL，默认会自动搜索。也可以显式传参：

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "input_text": "The company reported revenue of $10 billion in its 2024 annual report [1].\n\n[1] 2024 annual report revenue $10 billion",
    "enable_web_search": true,
    "max_search_results": 2
  }'
```

搜索发现的来源会标记为 `discovered_source`。它可以参与 claim 支撑判断，但不会被当作作者原文明确引用的上游来源。

## 运行测试

```bash
cd source_grounding_auditor
PYTHONPATH=backend pytest -q backend/tests
```

## 后续接入 LLM 的位置

- `backend/app/providers/llm_provider.py`
- `backend/app/claim_extractor.py`
- `backend/app/support_checker.py`

生产版本应要求 LLM 输出符合 `backend/app/schemas.py` 中的 Pydantic schema，并在进入 analyzer 前做校验。

## 后续接入搜索的规则

搜索结果只能作为 `discovered_source`，不能自动成为真实上游来源边。只有当一个 source 文本显式链接、引用或声明依赖另一个 source 时，才能创建 `upstream_source` edge。
