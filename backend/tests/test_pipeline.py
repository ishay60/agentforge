"""End-to-end: ingest 3 docs (txt, html, pdf) -> hybrid search -> persist -> reload."""

import pymupdf

from agentforge import retriever
from agentforge.ingestion import html_to_text, ingest
from agentforge.retriever import HybridRetriever


def test_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(retriever, "INDEX_ROOT", tmp_path / "idx")

    (tmp_path / "returns.txt").write_text(
        "Return policy: items may be returned within 30 days with a receipt. " * 20
    )
    (tmp_path / "pw.html").write_text(
        "<html><head><script>x()</script></head><body><nav>menu</nav>"
        "<main><h1>Password reset</h1><p>Click 'Forgot password' on the login page, "
        "then follow the emailed link to reset your password.</p></main></body></html>"
    )
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "Error code E-4012 means the payment gateway timed out.")
    pdf.new_page().insert_text((72, 72), "Shipping takes 3-5 business days.")
    pdf.save(tmp_path / "errors.pdf")

    assert "x()" not in html_to_text((tmp_path / "pw.html").read_text())
    assert "menu" not in html_to_text((tmp_path / "pw.html").read_text())

    r = HybridRetriever()
    hashes = set()
    for name in ("returns.txt", "pw.html", "errors.pdf"):
        h, chunks = ingest(str(tmp_path / name))
        assert chunks and all(len(c["content"]) <= 512 for c in chunks)
        hashes.add(h)
        r.add(chunks)
    assert len(hashes) == 3

    top, _ = r.search("how do I reset my password", top_k=1)[0]
    assert top["source"].endswith("pw.html")

    top, _ = r.search("E-4012", top_k=1, alpha=0.3)[0]  # exact-match: BM25 carries it
    assert top["source"].endswith("errors.pdf") and top["page"] == 1

    r.save("agent-1")
    r2 = HybridRetriever.load("agent-1")
    assert len(r2.chunks) == len(r.chunks)
    assert r2.search("shipping time", top_k=1)[0][0]["page"] == 2
    assert HybridRetriever.load("nope").search("x") == []
