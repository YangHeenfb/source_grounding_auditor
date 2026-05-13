from app.citation_parser import (
    citations_near_text_span,
    parse_citations,
    parse_reference_descriptions,
    reference_descriptions_near_text_span,
)
from app.schemas import EdgeBasis


def test_parse_markdown_and_raw_url():
    text = "The report says revenue grew [SEC filing](https://www.sec.gov/example). More context https://example.com/page."
    citations = parse_citations(text)
    urls = [c.url for c in citations]
    assert "https://www.sec.gov/example" in urls
    assert "https://example.com/page" in urls
    assert any(c.kind == EdgeBasis.MARKDOWN_CITATION for c in citations)


def test_parse_footnote_definition_and_aligns_reference():
    text = "Revenue was $10 billion [1].\n\n[1]: https://example.com/annual-report"
    citations = parse_citations(text)
    near = citations_near_text_span(text, "Revenue was $10 billion [1].", citations)
    assert any(c.url == "https://example.com/annual-report" for c in near)


def test_parse_reference_descriptions_without_urls():
    text = "Revenue was $10 billion [1].\n\n来源指针\n[1] 2024 annual report revenue $10 billion"
    references = parse_reference_descriptions(text)
    near = reference_descriptions_near_text_span(text, "Revenue was $10 billion [1].", references)
    assert len(references) == 1
    assert near[0].label == "1"
    assert "annual report" in near[0].description
