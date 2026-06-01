from types import SimpleNamespace

import httpx

from app import source_fetcher as source_fetcher_module
from app.schemas import AccessStatus
from app.source_fetcher import SourceFetcher


class TLSFailingClient:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url):
        raise httpx.ConnectError("[SSL: SSLV3_ALERT_HANDSHAKE_FAILURE] sslv3 alert handshake failure")


def test_cuhk_tls_failure_uses_curl_fallback(monkeypatch):
    monkeypatch.setattr(source_fetcher_module.httpx, "Client", TLSFailingClient)

    def fake_run(cmd, capture_output, text, check):
        assert cmd[0] == "curl"
        html = (
            "<html><title>CUHK Shenzhen</title><body>"
            "The Chinese University of Hong Kong, Shenzhen article body. "
            "AIRS artificial intelligence and robotics institute. 95.69% employment."
            "</body></html>"
        )
        return SimpleNamespace(returncode=0, stdout=f"{html}\n200", stderr="")

    monkeypatch.setattr(source_fetcher_module.subprocess, "run", fake_run)

    fetcher = SourceFetcher(enable_url_fetch=True, timeout=3)
    for url in [
        "https://www.cuhk.edu.cn/en/article/17290",
        "https://www.cuhk.edu.cn/en/article/4974",
    ]:
        source = fetcher.fetch_url(url, "s001", [])

        assert source.access_status == AccessStatus.ACCESSIBLE
        assert source.fetch_method == "curl_fallback"
        assert source.extracted_text
        assert "CUHK Shenzhen" in source.title
