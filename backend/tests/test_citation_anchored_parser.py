from app.citation_parser import parse_citations, parse_reference_descriptions
from app.citation_units import build_cited_statements, build_source_registry


def _statements(text: str):
    citations = parse_citations(text)
    references = parse_reference_descriptions(text)
    return build_cited_statements(text, citations, references)


def test_chinese_citations_after_sentence_punctuation_bind_to_own_statement():
    text = (
        "Apple 营收为 3910 亿美元。[1] Microsoft 营收为 2451 亿美元。[2]\n\n"
        "来源指针\n"
        '[1]: https://example.com/apple "Apple Source"\n'
        '[2]: https://example.com/microsoft "Microsoft Source"'
    )

    statements = _statements(text)

    assert len(statements) == 2
    assert statements[0].cited_text == "Apple 营收为 3910 亿美元。"
    assert [(edge.label, edge.source_url) for edge in statements[0].citation_edges] == [
        ("1", "https://example.com/apple")
    ]
    assert statements[1].cited_text == "Microsoft 营收为 2451 亿美元。"
    assert [(edge.label, edge.source_url) for edge in statements[1].citation_edges] == [
        ("2", "https://example.com/microsoft")
    ]


def test_chinese_citations_before_sentence_punctuation_exclude_marker_and_keep_punctuation():
    text = (
        "Apple 营收为 3910 亿美元[1]。Microsoft 营收为 2451 亿美元[2]。\n\n"
        "来源指针\n"
        '[1]: https://example.com/apple "Apple Source"\n'
        '[2]: https://example.com/microsoft "Microsoft Source"'
    )

    statements = _statements(text)

    assert len(statements) == 2
    assert statements[0].cited_text == "Apple 营收为 3910 亿美元。"
    assert statements[1].cited_text == "Microsoft 营收为 2451 亿美元。"
    assert statements[0].citation_edges[0].marker_text == "[1]"
    assert statements[1].citation_edges[0].marker_text == "[2]"


def test_adjacent_citation_cluster_creates_one_statement_with_multiple_edges():
    text = (
        "第一句。[1][2] 第二句。[3]\n\n"
        "来源指针\n"
        '[1]: https://example.com/one "One"\n'
        '[2]: https://example.com/two "Two"\n'
        '[3]: https://example.com/three "Three"'
    )

    statements = _statements(text)

    assert len(statements) == 2
    assert statements[0].cited_text == "第一句。"
    assert [edge.label for edge in statements[0].citation_edges] == ["1", "2"]
    assert [edge.source_url for edge in statements[0].citation_edges] == [
        "https://example.com/one",
        "https://example.com/two",
    ]
    assert statements[1].cited_text == "第二句。"
    assert [edge.label for edge in statements[1].citation_edges] == ["3"]


def test_no_whitespace_required_after_citation_marker():
    text = (
        "第一句。[1]第二句。[2]\n\n"
        "来源指针\n"
        '[1]: https://example.com/one "One"\n'
        '[2]: https://example.com/two "Two"'
    )

    statements = _statements(text)

    assert len(statements) == 2
    assert statements[0].cited_text == "第一句。"
    assert statements[0].citation_edges[0].label == "1"
    assert statements[1].cited_text == "第二句。"
    assert statements[1].citation_edges[0].label == "2"


def test_source_registry_only_text_builds_registry_but_no_cited_statements():
    text = '来源指针\n\n[1]: https://example.com "Example Source"'
    citations = parse_citations(text)
    references = parse_reference_descriptions(text)

    statements = build_cited_statements(text, citations, references)
    registry = build_source_registry(citations, references)

    assert statements == []
    assert registry["1"].url == "https://example.com"
    assert registry["1"].title == "Example Source"


def test_body_citation_uses_registry_url_and_registry_line_is_not_audited():
    text = '正文第一句。[1]\n\n来源指针\n\n[1]: https://example.com "Example Source"'
    statements = _statements(text)

    assert len(statements) == 1
    assert statements[0].citation_edges[0].source_url == "https://example.com"
    assert statements[0].cited_text == "正文第一句。"
    assert statements[0].citation_edges[0].source_title == "Example Source"


def test_source_pointer_entry_overrides_same_number_markdown_definition():
    text = (
        "BOTZ 费用率是 0.68%，资产规模约 37.4 亿美元。[2] "
        "AIQ 费用率同样是 0.68%，资产规模约 108.5 亿美元。[8]\n\n"
        "来源指针\n"
        "[2] BOTZ 的成立日期、费用率、资产规模和 NAV 来自 Global X 官方页面。([Global X ETFs][1])\n"
        "[8] AIQ 的费用率和资产规模来自 Global X 官方页面。([Global X ETFs][2])\n\n"
        '[1]: https://www.globalxetfs.com/funds/botz/ "BOTZ"\n'
        '[2]: https://www.globalxetfs.com/funds/aiq/ "AIQ"'
    )

    statements = _statements(text)

    assert len(statements) == 2
    assert statements[0].citation_edges[0].label == "2"
    assert statements[0].citation_edges[0].source_url == "https://www.globalxetfs.com/funds/botz/"
    assert statements[1].citation_edges[0].label == "8"
    assert statements[1].citation_edges[0].source_url == "https://www.globalxetfs.com/funds/aiq/"


def test_source_pointer_without_resolvable_url_does_not_fall_back_to_same_label_definition():
    text = (
        "BOTZ 当前价格约 40.56 美元。[7]\n\n"
        "来源指针\n"
        "[7] BOTZ 的当前价格来自实时行情工具，价格约 40.56 美元，时间为 2026 年 6 月 1 日。\n\n"
        '[7]: https://example.com/not-the-price-source "Wrong same-label definition"'
    )

    statements = _statements(text)
    registry = build_source_registry(parse_citations(text), parse_reference_descriptions(text))

    assert registry["7"].registry_type == "source_pointer_entry"
    assert registry["7"].url is None
    assert len(statements) == 1
    assert statements[0].citation_edges[0].source_url is None


def test_markdown_bold_marker_before_citation_is_not_cited_text():
    text = (
        "港中深毕业生拿的是 CUHK 学位，而且 CUHK 和 CUHK 深圳由同一个 Senate 监督，"
        "共享学术结构和课程设计。**[5]\n\n"
        "来源指针\n"
        '[5]: https://www.cuhk.edu.cn/en/article/4974 "CUHK Shenzhen Article"'
    )

    statements = _statements(text)

    assert len(statements) == 1
    assert statements[0].cited_text == (
        "港中深毕业生拿的是 CUHK 学位，而且 CUHK 和 CUHK 深圳由同一个 Senate 监督，"
        "共享学术结构和课程设计。"
    )
    assert statements[0].cited_text != "**"
    assert not statements[0].cited_text.endswith("**")
