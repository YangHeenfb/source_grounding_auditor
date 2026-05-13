# Source Grounding Auditor MVP 需求文档

版本：0.1
目标读者：Codex 或工程实现人员
产品定位：分析带来源内容的证据链支撑结构，而不是判断观点对错

## 1. 一句话目标

构建一个工具，输入一段 AI 生成回答、网络文章文本或带引用内容，系统自动拆分其中的原子论断，追踪每条论断的来源链，判断它最终落在公开可验证的一手事实依据、弱事实依据、观点或归属依据，还是不可公开验证或引用错配，并输出各类比例。

## 2. 核心原则

1. 工具不判断观点是否正确。

2. 工具只判断一个论断在多大程度上被公开可验证的事实依据支撑。

3. 标注了来源不等于有事实支撑。系统必须区分 source exists、source relevant、source supports claim、source is primary factual evidence 这四件事。

4. 对不可验证内容，工具不能说它一定是假的，只能标注为公开可审计性低、来源链断裂、匿名来源、笼统来源或引用错配。

5. MVP 不输出单一可信度分数，只输出比例、分类、证据链和高风险项。

## 3. MVP 范围

### 3.1 MVP 要支持的输入

1. 用户粘贴一段 AI 生成回答。

2. 用户粘贴一篇网络文章正文。

3. 用户粘贴带 Markdown citation、URL、脚注或参考文献的文本。

4. 可选输入原始问题，用于判断哪些 claim 是核心 claim。

### 3.2 MVP 暂不支持

1. 暂不处理 YouTube 视频、音频转写和字幕抽取。

2. 暂不承诺找到真正原始来源，只找当前可公开验证的最上游来源。

3. 暂不做最终真假判定。

4. 暂不做政治立场、价值观或伦理立场判断。

5. 暂不做复杂图数据库。MVP 可以用 SQLite 或 PostgreSQL 存储 claim、source 和 edge。

## 4. 关键定义

### 4.1 原子论断

原子论断是最小可单独审计的信息单位。一个句子可以包含多个原子论断。

例子：

原句：OpenAI 在 2024 年发布了 GPT 4o，并且它支持文本、音频和图像输入。

拆分为：

1. OpenAI 在 2024 年发布了 GPT 4o。

2. GPT 4o 支持文本输入。

3. GPT 4o 支持音频输入。

4. GPT 4o 支持图像输入。

### 4.2 公开可验证的最上游来源

在公开可访问范围内，能够直接承载或支持该论断，并且没有发现其依赖更早公开来源的上游材料。

常见例子：

1. 官方数据表。

2. 法律或法规原文。

3. 法院文件。

4. 公司财报或公告。

5. 原始论文或高质量证据综述。

6. 原始采访 transcript。

7. 原始视频、演讲记录或新闻发布会记录。

### 4.3 观点包装

观点包装是指一个 source 被作者或 AI 当作事实支撑使用，但追踪后发现它本质上只是观点、评论、预测、专家判断、社论、分析文章或二手解释。

### 4.4 引用错配

引用错配是指 source 存在，但不能支持对应 claim。

常见情况：

1. source 只支持相关性，但 claim 写成因果性。

2. source 只适用于特定人群，但 claim 写成普遍结论。

3. source 是旧数据，但 claim 写成当前事实。

4. source 是观点文章，但 claim 写成事实依据。

5. source 只是主题相关，但没有对应证据。

## 5. 用户端分类体系

用户界面只展示少量分类，避免类别过多导致难以比较。

### 5.1 Claim 类型

1. 事实陈述

事件、状态、定义、日期、地点、主体关系、功能描述。

2. 归属陈述

某人说过什么、某报告称什么、某研究发现什么、某机构发布什么。

3. 判断陈述

观点、因果、预测、建议、解释、价值判断、策略判断。

4. 非论断文本

修辞、过渡句、空泛表达、无法独立审计的句子。这个类别不进入核心比例，只用于覆盖率说明。

### 5.2 数据标签

数据不是单独主类，而是标签。

字段：has_quantitative_data: true 或 false

含义：claim 是否包含数字、比例、金额、排名、统计趋势、样本量或时间序列。

## 6. 输出分类体系

每个原子论断最终归入下面四个公开输出桶之一。

### 6.1 硬事实支撑

条件：找到公开可验证事实来源，并且 source 直接支持 claim。

例子：官方统计数据直接支持某个数字。法规原文直接支持某条规则。财报直接支持营收数字。

### 6.2 弱事实支撑

条件：找到事实性来源，但支持关系不完整、不直接或较弱。

例子：source 只支持一部分 claim。source 只支持更弱版本。source 是可信二手来源，但没有追到一手来源。source 有时效性风险。

### 6.3 归属或观点支撑

条件：source 能证明某人、某机构、某报告或某专家表达过某种说法，但不能把该说法当作事实证明。

例子：专家评论、分析文章、社论、预测报告、采访观点。

### 6.4 不可公开验证或引用错配

条件：来源笼统、匿名、不可访问、链条断裂，或者 source 与 claim 不对应。

例子：experts say 但没有专家姓名。according to sources 但无法公开审计。引用链接中没有对应内容。source 与 claim 相反。

## 7. 需要输出的比例

MVP 不输出单一可信度分数。输出以下比例。

### 7.1 内容构成比例

基于 atomic claims 计算。

1. 事实陈述比例。

2. 归属陈述比例。

3. 判断陈述比例。

4. 含数据 claim 比例。

### 7.2 来源链终点比例

基于可审计 claim 计算。

1. 硬事实支撑比例。

2. 弱事实支撑比例。

3. 归属或观点支撑比例。

4. 不可公开验证或引用错配比例。

### 7.3 支撑关系比例

内部可细分，前端可折叠展示。

1. 直接支持。

2. 部分支持。

3. 只支持较弱版本。

4. 归属成立但事实未证实。

5. 仅观点支撑。

6. 背景相关。

7. 不支持。

8. 矛盾。

9. 无法访问或无法判断。

### 7.4 核心产品指标

1. 公开事实支撑率 = 硬事实支撑比例。

2. 宽松事实支撑率 = 硬事实支撑比例 + 弱事实支撑比例。

3. 观点包装率 = 归属或观点支撑比例中，被原文当作事实依据使用的部分。

4. 来源不透明率 = 匿名、笼统、不可访问、链条断裂的比例。

5. 引用错配率 = source 存在但不支持、弱支持、背景相关或矛盾的比例。

## 8. 功能需求

### 8.1 文本解析

系统需要解析用户输入文本中的：

1. 段落。

2. URL。

3. Markdown citation。

4. 脚注形式 citation。

5. 参考文献列表。

6. 括号引用，例如作者年份。

### 8.2 Claim 抽取

系统需要将文本拆成 atomic claims，并输出结构化 JSON。

每条 claim 至少包含：

1. claim_id。

2. original_text_span。

3. normalized_claim。

4. claim_type: factual, attribution, judgment, non_claim。

5. has_quantitative_data。

6. source_mentions。

7. importance_label: core, supporting, background。

### 8.3 Citation 对齐

系统需要判断每条 claim 对应哪些 source。

对齐规则优先级：

1. 同句或同段落中的显式链接。

2. 脚注或引用编号。

3. 文末参考文献。

4. source mention，例如 according to WHO。

5. 如果没有显式 source，系统可以搜索候选来源，但必须标注为 discovered_source，而不是 author_cited_source。

### 8.4 Source 抽取

系统需要抓取或提取 source 内容。

每个 source 至少包含：

1. source_id。

2. url。

3. title。

4. publisher_or_author。

5. publication_date。

6. access_status: accessible, paywalled, failed, unavailable。

7. source_type: primary_fact_source, evidence_synthesis, secondary_reporting, opinion_analysis, anonymous_or_opaque, unknown。

8. extracted_text。

9. evidence_spans。

### 8.5 Source 支撑关系判断

系统需要判断 source 是否支持 claim。

支持关系枚举：

1. directly_supports。

2. partially_supports。

3. supports_weaker_claim。

4. attribution_only。

5. opinion_only。

6. background_only。

7. no_support。

8. contradicts。

9. inaccessible。

每个判断必须包含：

1. evidence_span。

2. reasoning_summary，简短说明，不暴露模型推理链。

3. final_bucket，映射到四个输出桶。

### 8.6 上游来源追踪

MVP 只递归追踪两层。

层级定义：

1. level 0: 用户输入文本。

2. level 1: 用户输入文本直接引用的 source。

3. level 2: level 1 source 明确引用、链接或声明依赖的上游 source。

停止条件：

1. 已找到 primary_fact_source 且直接支持 claim。

2. source 没有明确上游引用。

3. source 不可访问。

4. 已达到最大追踪深度。

5. 继续追踪会进入重复或循环引用。

重要规则：

系统不能把模型猜测的相关来源画成真实 source edge。只有在 source 中存在链接、参考文献、明确提及、数据来源声明或可验证 metadata 时，才创建真实上游 edge。

### 8.7 比例报告

系统需要输出：

1. 总 claim 数。

2. 可审计 claim 数。

3. 非论断文本数。

4. 内容构成比例。

5. 来源链终点比例。

6. 支撑关系比例。

7. 高风险 claim 列表。

8. 每个 claim 的证据链。

### 8.8 高风险 claim 列表

高风险 claim 包含：

1. judgment claim 依赖 opinion source，但被原文包装成事实。

2. source 与 claim 不对应。

3. source 只支持弱版本。

4. claim 使用匿名或笼统来源。

5. claim 含数字但未找到原始数据。

6. claim 是因果表述，但 source 只支持相关性。

## 9. 技术架构建议

### 9.1 MVP 推荐架构

后端：Python 3.11 + FastAPI

数据库：SQLite 起步，后续可换 PostgreSQL

前端：Next.js 或简单 React 单页应用

LLM：通过 provider interface 调用模型，优先支持 OpenAI Structured Outputs，也允许 mock provider 方便测试

搜索和抓取：先实现 provider interface，可接 Tavily、OpenAI web search、Firecrawl 或自定义搜索 API

图谱展示：前端先用简单列表和缩进树，后续再接 React Flow

### 9.2 目录结构建议

source_grounding_auditor/

1. backend/app/main.py

2. backend/app/models.py

3. backend/app/schemas.py

4. backend/app/claim_extractor.py

5. backend/app/citation_parser.py

6. backend/app/source_fetcher.py

7. backend/app/source_classifier.py

8. backend/app/support_checker.py

9. backend/app/upstream_tracer.py

10. backend/app/ratio_reporter.py

11. backend/app/providers/llm_provider.py

12. backend/app/providers/search_provider.py

13. backend/tests/test_claim_extraction.py

14. backend/tests/test_ratio_report.py

15. frontend/app/page.tsx

16. frontend/components/ClaimTable.tsx

17. frontend/components/RatioSummary.tsx

18. frontend/components/EvidenceChain.tsx

## 10. API 设计

### 10.1 POST /analyze

请求：

```json
{
  "input_text": "string",
  "original_question": "string optional",
  "mode": "ai_answer_or_article",
  "max_upstream_depth": 2
}
```

响应：

```json
{
  "analysis_id": "string",
  "summary": {
    "total_claims": 0,
    "auditable_claims": 0,
    "non_claim_items": 0,
    "content_mix": {
      "factual": 0.0,
      "attribution": 0.0,
      "judgment": 0.0,
      "has_quantitative_data": 0.0
    },
    "grounding_mix": {
      "hard_fact_grounding": 0.0,
      "weak_fact_grounding": 0.0,
      "attribution_or_opinion_grounding": 0.0,
      "unverifiable_or_mismatch": 0.0
    },
    "key_rates": {
      "public_fact_support_rate": 0.0,
      "loose_fact_support_rate": 0.0,
      "opinion_packaging_rate": 0.0,
      "source_opacity_rate": 0.0,
      "citation_mismatch_rate": 0.0
    }
  },
  "claims": []
}
```

### 10.2 GET /analysis/{analysis_id}

返回已完成分析。

### 10.3 GET /health

返回服务状态。

## 11. 数据模型

### 11.1 Claim

```json
{
  "claim_id": "c001",
  "original_text_span": "string",
  "normalized_claim": "string",
  "claim_type": "factual | attribution | judgment | non_claim",
  "has_quantitative_data": true,
  "importance_label": "core | supporting | background",
  "linked_source_ids": ["s001"],
  "final_bucket": "hard_fact_grounding | weak_fact_grounding | attribution_or_opinion_grounding | unverifiable_or_mismatch",
  "risk_flags": ["source_claim_mismatch", "causal_overclaim"],
  "evidence_chain": []
}
```

### 11.2 Source

```json
{
  "source_id": "s001",
  "url": "string",
  "title": "string",
  "publisher_or_author": "string",
  "publication_date": "string optional",
  "access_status": "accessible | paywalled | failed | unavailable",
  "source_type": "primary_fact_source | evidence_synthesis | secondary_reporting | opinion_analysis | anonymous_or_opaque | unknown",
  "extracted_text_preview": "string",
  "upstream_source_ids": ["s002"]
}
```

### 11.3 Evidence Edge

```json
{
  "claim_id": "c001",
  "source_id": "s001",
  "edge_type": "author_cited | discovered_source | upstream_source",
  "support_relation": "directly_supports | partially_supports | supports_weaker_claim | attribution_only | opinion_only | background_only | no_support | contradicts | inaccessible",
  "evidence_span": "string",
  "reasoning_summary": "string",
  "final_bucket": "hard_fact_grounding | weak_fact_grounding | attribution_or_opinion_grounding | unverifiable_or_mismatch"
}
```

## 12. LLM Structured Output Schemas

### 12.1 Claim Extraction Schema

```json
{
  "type": "object",
  "properties": {
    "claims": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "original_text_span": {"type": "string"},
          "normalized_claim": {"type": "string"},
          "claim_type": {"type": "string", "enum": ["factual", "attribution", "judgment", "non_claim"]},
          "has_quantitative_data": {"type": "boolean"},
          "source_mentions": {"type": "array", "items": {"type": "string"}},
          "importance_label": {"type": "string", "enum": ["core", "supporting", "background"]}
        },
        "required": ["original_text_span", "normalized_claim", "claim_type", "has_quantitative_data", "source_mentions", "importance_label"]
      }
    }
  },
  "required": ["claims"]
}
```

### 12.2 Support Check Schema

```json
{
  "type": "object",
  "properties": {
    "support_relation": {
      "type": "string",
      "enum": ["directly_supports", "partially_supports", "supports_weaker_claim", "attribution_only", "opinion_only", "background_only", "no_support", "contradicts", "inaccessible"]
    },
    "evidence_span": {"type": "string"},
    "reasoning_summary": {"type": "string"},
    "risk_flags": {"type": "array", "items": {"type": "string"}},
    "final_bucket": {
      "type": "string",
      "enum": ["hard_fact_grounding", "weak_fact_grounding", "attribution_or_opinion_grounding", "unverifiable_or_mismatch"]
    }
  },
  "required": ["support_relation", "evidence_span", "reasoning_summary", "risk_flags", "final_bucket"]
}
```

## 13. UI 要求

### 13.1 首页

包含：

1. 大文本框。

2. 可选原始问题输入框。

3. Analyze 按钮。

4. 示例输入按钮。

### 13.2 结果页

包含：

1. 内容构成比例卡片。

2. 来源链终点比例卡片。

3. 核心产品指标卡片。

4. Claim 表格。

5. 高风险 claim 列表。

6. 单条 claim 的证据链展开视图。

### 13.3 Claim 表格字段

1. Claim。

2. Claim 类型。

3. 是否含数据。

4. 对应 source。

5. 最上游公开来源。

6. 最终输出桶。

7. 风险标签。

8. 展开按钮。

## 14. 规则映射

### 14.1 映射到硬事实支撑

条件：

1. source_type 是 primary_fact_source 或 evidence_synthesis。

2. support_relation 是 directly_supports。

### 14.2 映射到弱事实支撑

条件：

1. source_type 是 primary_fact_source、evidence_synthesis 或 secondary_reporting。

2. support_relation 是 partially_supports 或 supports_weaker_claim。

### 14.3 映射到归属或观点支撑

条件：

1. support_relation 是 attribution_only 或 opinion_only。

2. 或 source_type 是 opinion_analysis。

### 14.4 映射到不可公开验证或引用错配

条件：

1. support_relation 是 background_only、no_support、contradicts 或 inaccessible。

2. 或 source_type 是 anonymous_or_opaque。

3. 或 access_status 是 failed 或 unavailable。

## 15. 风险标签

系统需要支持以下 risk_flags：

1. source_claim_mismatch。

2. causal_overclaim。

3. correlation_presented_as_causation。

4. outdated_source。

5. anonymous_source。

6. vague_source。

7. inaccessible_source。

8. opinion_used_as_fact。

9. secondary_source_only。

10. quantitative_claim_without_primary_data。

11. overgeneralization。

12. source_only_supports_weaker_claim。

## 16. 测试用例

### 16.1 测试一：明确事实来源

输入：

The company reported revenue of $10 billion in its 2024 annual report.

期望：

1. claim_type = factual。

2. has_quantitative_data = true。

3. 如果找到 annual report，final_bucket = hard_fact_grounding。

### 16.2 测试二：专家笼统说法

输入：

Experts say this policy will damage the middle class.

期望：

1. claim_type = judgment。

2. source_opacity flag。

3. final_bucket = unverifiable_or_mismatch，除非有具名专家和可访问 source。

### 16.3 测试三：观点来源包装成事实

输入：

A market commentary article shows that the company is guaranteed to dominate AI infrastructure.

期望：

1. claim_type = judgment。

2. source_type = opinion_analysis。

3. risk_flags 包含 opinion_used_as_fact。

4. final_bucket = attribution_or_opinion_grounding。

### 16.4 测试四：相关性被写成因果性

输入：

The study proves that coffee causes lower mortality.

如果 source 只说 association：

1. support_relation = supports_weaker_claim。

2. risk_flags 包含 correlation_presented_as_causation。

3. final_bucket = weak_fact_grounding。

### 16.5 测试五：source 与 claim 不对应

输入 claim：

The report says unemployment fell by 15 percent.

source 内容：

The report says inflation fell by 15 percent.

期望：

1. support_relation = no_support。

2. risk_flags 包含 source_claim_mismatch。

3. final_bucket = unverifiable_or_mismatch。

## 17. Codex 执行任务清单

### 17.1 第一阶段

1. 创建 FastAPI backend。

2. 创建 schemas.py，定义 Claim、Source、EvidenceEdge、AnalysisResult。

3. 实现 citation_parser.py，提取 URL、Markdown citation、脚注引用。

4. 实现 claim_extractor.py，先支持 mock mode，再支持 LLM provider。

5. 实现 ratio_reporter.py，根据 claim final_bucket 计算比例。

6. 写单元测试。

### 17.2 第二阶段

1. 实现 source_fetcher.py，抓取 URL 文本。

2. 实现 support_checker.py，使用 LLM 判断 claim 与 source 的关系。

3. 实现 source_classifier.py，判断 source_type。

4. 实现 upstream_tracer.py，追踪上游 source，最大深度为 2。

5. 实现 POST /analyze。

### 17.3 第三阶段

1. 创建前端页面。

2. 展示比例卡片。

3. 展示 claim 表格。

4. 展示证据链展开视图。

5. 展示高风险 claim 列表。

## 18. 验收标准

1. 用户可以粘贴一段文本并点击 Analyze。

2. 系统返回 atomic claims。

3. 系统能识别事实陈述、归属陈述、判断陈述和非论断文本。

4. 系统能解析文本中的 URL 和脚注。

5. 系统能为每条 claim 给出最终输出桶。

6. 系统能计算内容构成比例和来源链终点比例。

7. 系统不输出单一可信度分数。

8. 系统能列出高风险 claim。

9. 系统能在 mock mode 下通过全部测试。

10. 如果配置了 API key，系统可以调用真实 LLM 和搜索 provider。

## 19. 给 Codex 的直接执行提示

请根据本需求文档实现一个 MVP。优先实现 backend 和测试，再实现简单前端。不要追求完整产品，先保证以下功能可以跑通：输入文本，抽取 atomic claims，解析 citations，给出 claim 类型，模拟或真实判断 source 支撑关系，最后输出四类比例和高风险 claim 列表。

实现时请遵守：

1. 不输出可信度总分。

2. 不判断观点对错。

3. 不把搜索得到的相关来源当作真实上游引用边。

4. 每个 evidence edge 必须标明 basis，例如 explicit_link、footnote、reference_list、source_statement、discovered_source。

5. 所有 LLM 输出必须通过 schema 校验。

6. 没有 API key 时，mock mode 必须仍然可运行。

