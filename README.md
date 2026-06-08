# Source Grounding Auditor

**看一段带 citation 的内容，最终有多少落到事实，有多少停在观点。**

Source Grounding Auditor 是一个 citation terminal audit 工具。它不判断一篇文章或一段 AI 回答到底“对不对”，也不输出单一可信度分数。它只做一件更具体的事：

**把文本里的 citation 一层层往下追，看看它最终落到良好定义的事实来源，还是停在观点、评论、帖子、博主判断，或者根本无法审计。**

## 为什么做这个项目

现在大量 AI 生成回答、网络文章、研究笔记、YouTube 视频脚本和网页内容都会标注 citation。citation 会给人一种很强的安全感：一个论断只要旁边有来源，就好像已经被事实支持了。

但现实不是这样。

有些 citation 的确指向良好定义的事实，比如官方公告、原始数据、论文、财报、法律文本、公司页面、报告中的具体数据或访谈原文。

也有很多 citation 只是指向别人的观点，比如一篇博客、一条推文、一个论坛帖子、一篇评论文章、一个专家判断或一个投资分析。它们可能有价值，但它们不是事实本身。

更麻烦的是，观点也可以继续引用别的来源。一个博主的观点可能引用了官方数据，也可能引用了另一个博主的观点。Source Grounding Auditor 想回答的就是这个问题：

**这条 citation 继续往下追，最后到底落到了事实，还是仍然只是观点？**

## 这个项目不做什么

本项目不判断观点是否正确。

本项目不判断整篇文章是真还是假。

本项目不把“无法审计”当成“错误”。

本项目不输出“可信度 82 分”这类单一分数。

本项目不把搜索到的相似网页自动当成真实上游来源。

它的目标是给用户一个直观的证据结构视图：

**这段内容里，多少 citation 最终落到事实，多少 citation 最终停在观点，多少 citation 目前无法审计。**

## 核心概念

### Citation 不是证据，citation 只是入口

一条 citation 只能说明“作者给了一个来源”。它还不能说明这个来源真的支持原文，也不能说明这个来源本身是事实来源。

Source Grounding Auditor 把 citation 看成入口，然后继续追踪：

```text
原文中的 cited statement
→ citation
→ 直接来源
→ 来源中的证据片段
→ 来源自己引用的上游来源
→ 最终落点
```

最终落点只分成少数几类，方便用户一眼看懂。

### FACT：事实终点

citation 最终落到良好定义的事实来源。

例子包括官方页面、原始数据、法律法规、公司公告、财报、学术论文、报告中的具体数据、访谈原文、演讲 transcript、产品文档、基金官网和学校官网等。

这里的 FACT 不等于“宇宙真理”。它的意思是：

**这条 citation 最终落到了一个可以公开核验的事实性来源。**

### OPINION：观点终点

citation 最终停在观点、评论、博客、媒体分析、专家判断、投资建议、社交媒体帖子或价值判断，而且继续追溯后没有落到事实来源。

这不代表观点一定错。它只代表：

**这条 citation 最终没有落到良好定义的事实来源，用户需要自己判断这个观点是否成立。**

### UNRESOLVED：无法审计

citation 没有 URL、source body 缺失、网页抓取失败、citation UI 在复制文本时丢失，或者本轮没有足够信息判断最终落点。

这不代表 citation 错，也不代表内容错。它只表示：

**当前系统无法完成这条 citation 的证据链追溯。**

### MISMATCH：引用不对应

source 可访问，也有相关片段，但 source 明显不支持原文中的 cited statement，或者与原文相矛盾。

这个类别会作为 warning badge 显示，不放进主饼图里混淆用户。

## 最终用户看到什么

第一版界面只展示三个核心结果。

```text
事实终点：xx%
观点终点：xx%
无法审计：xx%
```

如果存在明显引用问题，再显示一个额外提醒：

```text
引用不对应：n 条
```

主界面不默认展示所有内部标签，不展示每条 claim 的 debug 信息，也不要求用户读一长串 support relation、risk flag 或 source opacity。

用户首先看到的是一个饼图和一个聚合证据树。

```text
文档
→ citation group
→ source
→ upstream source
→ FACT / OPINION / UNRESOLVED
```

点击某个 source 或终点节点时，才展开具体 cited text。

## Demo screenshot

下面是一段完整财经分析文本的展示结果：既有落到财报数据的事实终点，也有停在投资评论的观点终点，并保留少量无法审计和引用错配 warning。

![Citation terminal audit finance demo](docs/assets/citation-terminal-finance-paragraph-demo.png)

## 一个简单例子

原文写：

```text
BOTZ 的费用率是 0.68%。[1]
```

如果 `[1]` 指向 Global X 的 BOTZ 官方页面，且页面确实列出 0.68% 费用率，那么这条 citation 落到 FACT。

原文写：

```text
某博主认为 BOTZ 是最适合长期持有的机器人 ETF。[2]
```

如果 `[2]` 只是一个博客观点，并且博客没有继续引用事实来源，那么这条 citation 落到 OPINION。

如果博客继续引用了官方基金页面、持仓数据和费用率页面，系统会继续追溯。若最终能落到这些事实来源，它可以被标记为由事实支撑的观点链路。

如果 `[2]` 没有 URL，或者 source card 在复制文本时丢失，那么它落到 UNRESOLVED。

## 当前产品范围

它只分析已经标注 citation 的内容。未标注 citation 的段落不进入主比例统计。

我们想解决一个明确的问题：

**已经标注了来源的内容，来源到底最终落到事实还是观点？**

## 输入方式

当前后端支持两类输入。

### 纯文本输入

用户粘贴带 citation 的文章、AI 回答或研究笔记。

系统会尝试解析文本中可见的 citation marker，例如 `[1]`、Markdown link、URL 和 reference list。

纯文本模式只是 fallback。它无法保证捕获原网页里的 hidden source card、hover citation、侧边栏来源或 DOM only citation。

### 结构化 citation 输入

浏览器插件、API 客户端或上游系统可以直接传入结构化 citation。

例如：

```json
{
  "input_mode": "browser_dom",
  "dom_citations": [
    {
      "citation_id": "c1",
      "marker_text": "[1]",
      "source_url": "https://example.com/source",
      "source_title": "Example Source",
      "cited_text_span": "The company reported revenue of $10 billion.",
      "char_start": 120,
      "char_end": 168,
      "capture_method": "dom_anchor",
      "confidence": "high"
    }
  ]
}
```

结构化 citation 输入优先于纯文本 parser。未来浏览器插件会主要使用这种方式。

## 处理流程

```text
输入文本或结构化 citation
→ 定位 cited statements
→ 绑定 citation 和 source
→ 抓取 source body
→ 提取证据片段
→ 判断 source 是否支撑 cited statement
→ 如果 source 是观点，继续追踪它显式引用的上游来源
→ 得到最终落点：FACT / OPINION / UNRESOLVED / MISMATCH
→ 输出饼图和聚合证据树
```

系统只承认显式来源边。也就是说，只有当 source 文本中明确出现 URL、reference、citation 或上游来源声明时，才会继续追踪。系统不会因为语义相似就猜测上游来源。

## 当前能力

1. 输入 AI 回答、文章正文或带 citation 的文本。

2. 默认使用 citation only mode，只分析带 citation 的内容。

3. 支持 URL、Markdown citation、脚注和 reference list。

4. 支持结构化 `dom_citations` 和 `citation_annotations` 输入。

5. 支持 `provided_sources`，可以把 citation URL 映射到用户提供的 source text。

6. 把 citation 最终落点分为 FACT、OPINION、UNRESOLVED、MISMATCH。

7. 主界面用饼图展示事实终点、观点终点和无法审计。

8. 引用错配作为 warning badge 单独显示。

9. 输出文档级聚合证据树：document → citation group → source → terminal class。

10. 内部仍保留 claim、support relation、risk flags 等 debug 字段，但默认不作为用户主界面展示。

## 当前限制

1. 本项目不是严格意义上的 fact checker。

2. source support check 依赖结构化 LLM 判断，不能替代人工核查。

3. 纯文本输入无法捕获隐藏在网页 UI 里的 citation。

4. 默认抓取显式 URL，但某些网站可能因 TLS、反爬、动态渲染或登录墙抓取失败。

5. 没有 URL 的 citation 可能会落到 UNRESOLVED。

6. 上游来源追踪只承认显式 citation edge，不根据语义相似度猜测来源链。

7. MISMATCH 只表示 citation 与 cited statement 不对应，不代表整篇文章错误。

## 推荐产品形态

当前 Web UI 适合作为开发和测试入口。

更理想的真实使用方式是浏览器插件：

```text
用户打开 AI 回答、网页文章或研究笔记
→ 点击插件按钮
→ 插件从 DOM 捕获 citation、source URL 和 cited text span
→ 后端生成 FACT / OPINION / UNRESOLVED 比例和证据树
```

插件的价值在于减少纯文本复制带来的 citation 丢失。它不会替代后端审计引擎，只负责更可靠地捕获 citation 结构。

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

## 运行测试

```bash
cd source_grounding_auditor
PYTHONPATH=backend pytest -q backend/tests
```

## 后续方向

1. 浏览器插件捕获 DOM citation。

2. 更强的 source body 抓取和 fallback。

3. 更强的 evidence snippet retrieval。

4. 观点 source 的显式上游引用追踪。

5. 更清晰的聚合证据树交互。

6. 针对 AI 回答页面、网页文章和 YouTube 描述区的 capture adapter。

## 一句话总结

**Source Grounding Auditor 不告诉你观点对不对。它告诉你：这篇内容的 citation 最终落到事实，还是停在观点。**
