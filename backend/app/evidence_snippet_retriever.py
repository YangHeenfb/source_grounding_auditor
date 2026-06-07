from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceSnippet:
    text: str
    score: float
    basis: list[str]


@dataclass(frozen=True)
class SnippetRetrievalResult:
    snippets: list[EvidenceSnippet]
    failure_reason: str = ""
    status: str = "no_relevant_snippet"
    retrieval_query: str = ""
    candidate_count: int = 0


SENTENCE_RE = re.compile(r"[^。！？.!?\n]+(?:[。！？.!?]+|$)")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|:1|人|名|个|所|位|million|billion|%)?", re.IGNORECASE)
LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&.-]{1,}")
CAPITALIZED_PHRASE_RE = re.compile(r"\b(?:[A-Z][A-Za-z&.-]+(?:\s+|$)){1,5}")
CJK_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
MIN_SCORE = 1.0
FINANCE_SEMANTIC_HINTS = {
    "股票": ["stock", "stocks", "equity", "security"],
    "股东": ["stockholder", "shareholder"],
    "公司所有权": ["ownership in a company", "share of ownership"],
    "所有权": ["ownership", "share of ownership"],
    "资本增值": ["capital appreciation"],
    "分红": ["dividend", "dividend payments"],
    "股息": ["dividend", "dividend payments"],
    "利润": ["earnings", "profit"],
    "普通股": ["common stock", "common stockholders"],
    "优先股": ["preferred stock", "preferred stockholders"],
    "债权人": ["bondholders", "creditors"],
    "破产清算": ["bankrupt", "liquidation", "assets are liquidated"],
    "破产": ["bankrupt", "bankruptcy"],
    "剩余资产": ["proceeds", "what is left", "whatever is left"],
    "成交价格": ["last-traded price", "execution price"],
    "成交": ["execute", "execution", "last-traded price"],
    "市价单": ["market order"],
    "限价单": ["limit order"],
    "未来收益": ["future payoffs", "future income"],
    "折现": ["discounted value", "present value"],
    "风险回报": ["risk premium", "expected return"],
    "估值": ["asset valuations", "discounted value", "future payoffs"],
}
TITLE_SEMANTIC_HINTS = {
    "order types": [
        "market order",
        "limit order",
        "last-traded price",
        "execution price",
        "best available price",
        "real time quote",
    ],
    "stocks": [
        "share of ownership",
        "ownership in a company",
        "capital appreciation",
        "dividend payments",
    ],
    "stock liquidation": [
        "common stockholders",
        "preferred stockholders",
        "bondholders",
        "whatever is left",
        "last in line",
        "assets are liquidated",
    ],
    "dividend": [
        "portion of a company's profit",
        "paid to shareholders",
        "dividend payments",
    ],
    "asset valuations": [
        "expected discounted value",
        "future payoffs",
        "expected rate of return",
        "expected returns",
        "risk premium",
        "risk compensation",
        "expected future income",
    ],
}
SEMANTIC_PHRASES = {
    phrase.lower()
    for phrases in [*FINANCE_SEMANTIC_HINTS.values(), *TITLE_SEMANTIC_HINTS.values()]
    for phrase in phrases
    if " " in phrase
}


def retrieve_evidence_snippets(
    cited_text: str,
    source_text: str,
    *,
    top_k: int = 5,
    source_pointer_description: str = "",
    source_title: str = "",
    source_url: str = "",
) -> list[EvidenceSnippet]:
    return retrieve_evidence_snippets_with_reason(
        cited_text,
        source_text,
        top_k=top_k,
        source_pointer_description=source_pointer_description,
        source_title=source_title,
        source_url=source_url,
    ).snippets


def retrieve_evidence_snippets_with_reason(
    cited_text: str,
    source_text: str,
    *,
    top_k: int = 5,
    source_pointer_description: str = "",
    source_title: str = "",
    source_url: str = "",
) -> SnippetRetrievalResult:
    retrieval_query = build_retrieval_query(
        cited_text,
        source_pointer_description=source_pointer_description,
        source_title=source_title,
        source_url=source_url,
    )
    query = _features(retrieval_query)
    cleaned_source = _clean_source_text(source_text)
    if not cleaned_source.strip():
        return SnippetRetrievalResult([], "no_source_text", "no_relevant_snippet", retrieval_query, 0)
    if not query["all"]:
        candidates = _broad_candidates(
            cleaned_source,
            retrieval_query=retrieval_query,
            source_title=source_title,
            source_url=source_url,
            top_k=max(top_k, 20),
        )
        return SnippetRetrievalResult(
            candidates[:top_k],
            "semantic_rerank_needed",
            "semantic_rerank_needed" if candidates else "no_relevant_snippet",
            retrieval_query,
            len(candidates),
        )

    snippets: list[EvidenceSnippet] = []
    scored_candidates: list[EvidenceSnippet] = []
    for sentence in _sentences(cleaned_source):
        if not _contentful_sentence(sentence):
            continue
        if _title_like_sentence(sentence, source_title):
            continue
        sentence_features = _features(sentence)
        score = 0.0
        basis: list[str] = []

        number_overlap = query["numbers"] & sentence_features["numbers"]
        if number_overlap:
            score += 5.0 * len(number_overlap)
            basis.append(f"number overlap: {', '.join(sorted(number_overlap))}")

        entity_overlap = query["entities"] & sentence_features["entities"]
        if entity_overlap:
            score += 3.0 * len(entity_overlap)
            basis.append(f"entity/proper-noun overlap: {', '.join(sorted(entity_overlap)[:6])}")

        token_overlap = query["tokens"] & sentence_features["tokens"]
        if token_overlap:
            score += 1.0 * len(token_overlap)
            basis.append(f"token overlap: {', '.join(sorted(token_overlap)[:8])}")

        thematic_overlap = query["themes"] & sentence_features["themes"]
        if thematic_overlap:
            score += 3.0 * len(thematic_overlap)
            basis.append(f"theme overlap: {', '.join(sorted(thematic_overlap)[:8])}")

        phrase_overlap = query["phrases"] & sentence_features["phrases"]
        if phrase_overlap:
            score += 6.0 * len(phrase_overlap)
            basis.append(f"semantic phrase overlap: {', '.join(sorted(phrase_overlap)[:6])}")

        if query["holdings_intent"] and sentence_features["holdings_terms"]:
            score += 4.0
            basis.append("holdings context")

        if query["fee_intent"] and sentence_features["fee_terms"]:
            score += 3.0
            basis.append("fee context")

        if query["liquidation_intent"] and sentence_features["liquidation_terms"]:
            score += 6.0
            basis.append("liquidation/common-stock context")

        liquidation_token_overlap = (
            {"bondholders", "creditors", "preferred", "common", "left"}
            & query["tokens"]
            & sentence_features["tokens"]
        )
        if query["liquidation_intent"] and liquidation_token_overlap:
            score += 4.0 * len(liquidation_token_overlap)
            basis.append(f"liquidation term overlap: {', '.join(sorted(liquidation_token_overlap))}")

        if score > 0:
            candidate = EvidenceSnippet(text=sentence, score=score, basis=basis or ["low lexical overlap"])
            scored_candidates.append(candidate)
            if score >= MIN_SCORE:
                snippets.append(candidate)

    snippets.sort(key=lambda item: item.score, reverse=True)
    if snippets:
        return SnippetRetrievalResult(snippets[:top_k], "", "lexical_match", retrieval_query, len(snippets))
    if scored_candidates:
        scored_candidates.sort(key=lambda item: item.score, reverse=True)
        return SnippetRetrievalResult(
            scored_candidates[: max(top_k, 8)],
            "candidate_snippets_low_score",
            "semantic_rerank_needed",
            retrieval_query,
            len(scored_candidates),
        )
    broad = _broad_candidates(
        cleaned_source,
        retrieval_query=retrieval_query,
        source_title=source_title,
        source_url=source_url,
        top_k=max(top_k, 20),
    )
    if broad:
        return SnippetRetrievalResult(
            broad[: max(top_k, 8)],
            "semantic_rerank_needed",
            "semantic_rerank_needed",
            retrieval_query,
            len(broad),
        )
    return SnippetRetrievalResult([], "no_feature_overlap", "no_relevant_snippet", retrieval_query, 0)


def build_retrieval_query(
    cited_text: str,
    *,
    source_pointer_description: str = "",
    source_title: str = "",
    source_url: str = "",
) -> str:
    cleaned_cited = clean_markdown_boundaries(cited_text)
    domain = _domain(source_url)
    parts = [cleaned_cited, source_pointer_description or "", source_title or "", domain]
    expanded = " ".join(part for part in parts if part)
    hints: list[str] = []
    haystack = expanded.lower()
    for zh_term, english_terms in FINANCE_SEMANTIC_HINTS.items():
        if zh_term in expanded:
            hints.extend(english_terms)
    for title_term, english_terms in TITLE_SEMANTIC_HINTS.items():
        if title_term == "stock liquidation":
            continue
        if title_term in haystack:
            hints.extend(english_terms)
    if any(term in expanded for term in ["普通股", "优先股", "债权人", "破产", "清算", "剩余资产"]) or any(
        term in haystack for term in ["common stock", "preferred stock", "bondholders", "bankrupt", "liquidated"]
    ):
        hints.extend(TITLE_SEMANTIC_HINTS["stock liquidation"])
    if "federalreserve.gov" in haystack:
        hints.extend(["asset valuations", "discounted value", "future payoffs", "risk premium", "risk compensation"])
    return " ".join(dict.fromkeys([*parts, *hints]))


def clean_markdown_boundaries(value: str) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    for marker in ("**", "__", "*", "_", "`"):
        while text.startswith(marker):
            text = text[len(marker):].lstrip()
        while text.endswith(marker):
            text = text[: -len(marker)].rstrip()
    return text.strip()


def _sentences(text: str) -> list[str]:
    pieces = [re.sub(r"\s+", " ", match.group(0)).strip() for match in SENTENCE_RE.finditer(text)]
    expanded: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        words = piece.split()
        if len(piece) > 700 and len(words) > 80:
            window = 80
            stride = 40
            for start in range(0, len(words), stride):
                chunk = " ".join(words[start : start + window]).strip()
                if chunk:
                    expanded.append(chunk)
                if start + window >= len(words):
                    break
        elif len(piece) > 700:
            window_chars = 360
            stride_chars = 180
            for start in range(0, len(piece), stride_chars):
                chunk = piece[start : start + window_chars].strip()
                if chunk:
                    expanded.append(chunk)
                if start + window_chars >= len(piece):
                    break
        else:
            expanded.append(piece)
    return [piece for piece in expanded if piece]


def _broad_candidates(
    text: str,
    *,
    retrieval_query: str = "",
    source_title: str = "",
    source_url: str = "",
    top_k: int = 20,
) -> list[EvidenceSnippet]:
    sentences = [sentence for sentence in _sentences(text) if _contentful_sentence(sentence)]
    title_features = _features(" ".join([retrieval_query, source_title, _domain(source_url)]))
    title_tokens = (
        title_features["tokens"]
        | title_features["entities"]
        | title_features["themes"]
        | title_features["phrases"]
    )
    candidates: list[EvidenceSnippet] = []
    seen: set[str] = set()

    for sentence in sentences:
        if _title_like_sentence(sentence, source_title):
            continue
        sentence_features = _features(sentence)
        overlap = title_tokens & (
            sentence_features["tokens"]
            | sentence_features["entities"]
            | sentence_features["themes"]
            | sentence_features["phrases"]
        )
        if overlap and sentence not in seen:
            seen.add(sentence)
            candidates.append(EvidenceSnippet(sentence, 0.5 + len(overlap), [f"title/domain overlap: {', '.join(sorted(overlap)[:6])}"]))

    for sentence in sentences[:top_k]:
        if sentence in seen:
            continue
        seen.add(sentence)
        candidates.append(EvidenceSnippet(sentence, 0.0, ["broad main-content candidate"]))
        if len(candidates) >= top_k:
            break
    return candidates[:top_k]


def _contentful_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    if len(sentence) < 35:
        return bool(NUMBER_RE.search(sentence) or CJK_TOKEN_RE.search(sentence))
    noisy = {
        "skip to main content",
        "official website",
        "here’s how you know",
        "privacy",
        "accessibility",
        "menu",
        "calculator",
        "auxiliary header",
        "featured content",
        "jumpstart your child",
        "financial tools",
        "learn about tax-advantaged",
        "take this month",
        "微信扫一扫",
        "复制链接",
        "我的收藏",
        "本页目录",
        "版权所有",
        "售前咨询",
        "售后服务",
        "工单提交",
        "建议反馈",
        "待支付订单",
        "待续费产品",
        "最近搜索",
        "热门搜索",
        "文档反馈官",
        "文档活动",
        "上一篇",
        "下一篇",
        "全部产品",
        "搜索本产品",
        "检测到您已登录",
        "国际站账号",
        "mainBtnText",
        "productIcon",
        "res-static",
        "设为首页",
        "加入收藏",
        "当前位置",
        "【打印】",
        "【纠错】",
        "服务声明",
        "产品动态",
        "新手指引",
        "新手入门",
        "操作导航",
        "DeepSeek",
    }
    return not any(needle in lowered for needle in noisy)


def _title_like_sentence(sentence: str, source_title: str) -> bool:
    if not source_title:
        return False
    sentence_norm = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
    title_norm = re.sub(r"[^a-z0-9]+", " ", source_title.lower()).strip()
    if not sentence_norm or not title_norm:
        return False
    if sentence_norm == title_norm:
        return True
    title_words = {word for word in title_norm.split() if len(word) > 2}
    sentence_words = sentence_norm.split()
    if len(sentence_words) <= 8 and len(title_words) >= 2 and title_words.issubset(set(sentence_words)):
        return True
    return False


def _features(text: str) -> dict[str, set[str]]:
    normalized_text = text or ""
    normalized_lower = normalized_text.lower()
    numbers = {re.sub(r"\s+", "", item.lower()) for item in NUMBER_RE.findall(normalized_text)}
    latin = {item.lower().strip(".") for item in LATIN_TOKEN_RE.findall(text or "") if len(item.strip(".")) >= 2}
    entities = {
        re.sub(r"\s+", " ", item).strip().lower()
        for item in CAPITALIZED_PHRASE_RE.findall(text or "")
        if item.strip()
    }
    cjk = {item for item in CJK_TOKEN_RE.findall(text or "") if len(item) >= 2}
    cjk_grams = _cjk_ngrams(text or "")
    stopwords = {
        "the",
        "and",
        "or",
        "of",
        "in",
        "to",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "with",
        "for",
        "page",
        "source",
    }
    tokens = {token for token in latin if token not in stopwords}
    tokens.update(cjk)
    tokens.update(cjk_grams)
    themes = _themes(normalized_lower)
    phrases = _phrases(normalized_lower)
    holdings_intent = _has_any(normalized_lower, {"前十大", "持仓", "权重", "合计", "接近", "集中度", "top holdings", "holdings", "weight"})
    fee_intent = _has_any(normalized_lower, {"费用率", "费率", "expense ratio", "management fee", "0.68"})
    liquidation_intent = _has_any(
        normalized_lower,
        {"普通股", "优先股", "债权人", "破产", "清算", "剩余资产", "common stock", "preferred stock", "bondholders", "bankrupt", "liquidated", "left"},
    )
    holdings_terms = _has_any(normalized_lower, {"holdings", "weight", "top", "keyence", "fanuc", "abb", "nvidia", "intuitive", "yaskawa", "smc", "daifuku", "inovance"})
    fee_terms = _has_any(normalized_lower, {"expense ratio", "management fee", "gross expense", "net expense", "0.68", "费用率"})
    liquidation_terms = _has_any(
        normalized_lower,
        {"common stockholder", "common stockholders", "preferred stockholder", "preferred stockholders", "bondholders", "bankrupt", "assets are liquidated", "whatever is left", "last in line"},
    )
    all_features = numbers | entities | tokens | themes
    return {
        "numbers": numbers,
        "entities": entities,
        "tokens": tokens,
        "themes": themes,
        "phrases": phrases,
        "holdings_intent": {"holdings_intent"} if holdings_intent else set(),
        "fee_intent": {"fee_intent"} if fee_intent else set(),
        "liquidation_intent": {"liquidation_intent"} if liquidation_intent else set(),
        "holdings_terms": {"holdings_terms"} if holdings_terms else set(),
        "fee_terms": {"fee_terms"} if fee_terms else set(),
        "liquidation_terms": {"liquidation_terms"} if liquidation_terms else set(),
        "all": all_features,
    }


def _themes(text: str) -> set[str]:
    theme_map = {
        "holdings": {"前十大", "持仓", "权重", "集中度", "top holdings", "holdings", "weight", "keyence", "fanuc", "abb", "nvidia"},
        "fees": {"费用率", "费率", "expense ratio", "management fee", "0.68"},
        "country": {"国家", "美国", "日本", "中国", "瑞士", "country", "geographic", "japan", "china", "switzerland"},
        "risk": {"风险", "波动", "beta", "standard deviation", "volatility", "标准差"},
    }
    return {theme for theme, needles in theme_map.items() if _has_any(text, needles)}


def _phrases(text: str) -> set[str]:
    return {phrase for phrase in SEMANTIC_PHRASES if phrase in text}


def _cjk_ngrams(text: str) -> set[str]:
    grams: set[str] = set()
    for run in CJK_TOKEN_RE.findall(text or ""):
        if len(run) < 2:
            continue
        for size in (2, 3, 4):
            if len(run) < size:
                continue
            for index in range(0, len(run) - size + 1):
                grams.add(run[index : index + size])
    noisy = {
        "腾讯",
        "腾讯云",
        "华为",
        "华为云",
        "阿里",
        "阿里云",
        "产品",
        "服务",
        "文档",
        "说明",
        "官方",
        "页面",
        "可以",
        "通过",
        "查看",
        "搜索",
        "中心",
    }
    return {gram for gram in grams if gram not in noisy}


def _has_any(text: str, needles: set[str]) -> bool:
    return any(needle in text for needle in needles)


def _clean_source_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""
    noisy_phrases = [
        "Skip to content",
        "Search",
        "Menu",
        "Privacy Policy",
        "Terms of Use",
        "Cookie",
        "Subscribe",
    ]
    for phrase in noisy_phrases:
        cleaned = cleaned.replace(phrase, " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _domain(url: str) -> str:
    if not url:
        return ""
    if "://" in url:
        return url.split("://", 1)[1].split("/", 1)[0]
    return url.split("/", 1)[0]
