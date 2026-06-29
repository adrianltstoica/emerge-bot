import base64
import importlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def import_test_app(monkeypatch, tmp_path):
    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text(
        json.dumps(
            [
                {
                    "text": "[Source: EMERGE Test]\nTrust and transparency matter for AI ethics.",
                    "source": "EMERGE Test",
                    "chunk_id": 0,
                },
                {
                    "text": "[Source: EU Test]\nHigh-risk AI systems require oversight.",
                    "source": "EU Test",
                    "chunk_id": 0,
                },
            ]
        ),
        encoding="utf-8",
    )
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    source_metadata_file = tmp_path / "source_metadata.json"
    source_metadata_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": [
                    {
                        "source_id": "EMERGE Test",
                        "citation": "EMERGE Test Source",
                        "title": "EMERGE Test Source Title",
                        "year": "2026",
                        "source_tier": "Core EMERGE deliverable",
                    },
                    {
                        "source_id": "Metadata Only",
                        "citation": "Metadata Only Source",
                        "title": "Metadata Only Source Title",
                        "year": "2025",
                        "source_tier": "Adjacent literature",
                        "text_extractable": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHUNKS_FILE", str(chunks_file))
    monkeypatch.setenv("DOCUMENTS_DIR", str(documents_dir))
    monkeypatch.setenv("VECTOR_INDEX_FILE", str(tmp_path / "vector_index.json.gz"))
    monkeypatch.setenv("SOURCE_METADATA_FILE", str(source_metadata_file))
    monkeypatch.setenv("EVALUATION_FILE", str(ROOT / "evaluation_questions.json"))
    monkeypatch.setenv("CHAT_LOG_DB", str(tmp_path / "chat_logs.db"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("ADMIN_SESSION_SECRET", raising=False)
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def read_static_index():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


def test_embedding_text_is_truncated_for_index_build():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import build_vector_index

        text = "x" * 13000
        assert len(build_vector_index.prepare_embedding_text(text, limit=12000)) == 12000
        assert build_vector_index.prepare_embedding_text("short", limit=12000) == "short"
    finally:
        sys.path.remove(str(ROOT / "scripts"))


def test_status_reports_loaded_corpus_and_tfidf_fallback(monkeypatch, tmp_path):
    app_module = import_test_app(monkeypatch, tmp_path)
    client = app_module.app.test_client()

    response = client.get("/status")

    assert response.status_code == 200
    data = response.get_json()
    assert data["ready"] is True
    assert data["chunks_loaded"] > 0
    assert data["indexed_sources"] > 0
    assert data["retrieval_backend"] == "tfidf"
    assert data["vector_index"] in {"missing", "chunk_count_mismatch", "loaded"}
    assert data["vector_index_file"]
    assert "vector_index_error" in data
    assert data["openai_api_key_configured"] is False
    assert data["anthropic_model"]
    assert data["anthropic_expansion_model"]
    assert data["source_metadata"] == "loaded"
    assert data["source_metadata_count"] == 2


def test_sources_endpoint_uses_metadata(monkeypatch, tmp_path):
    app_module = import_test_app(monkeypatch, tmp_path)
    client = app_module.app.test_client()

    response = client.get("/sources")

    assert response.status_code == 200
    data = response.get_json()
    assert data["metadata_loaded"] is True
    source = next(item for item in data["sources"] if item["source_id"] == "EMERGE Test")
    assert source["citation"] == "EMERGE Test Source"
    assert source["title"] == "EMERGE Test Source Title"
    assert source["chunk_count"] == 1
    metadata_only = next(item for item in data["sources"] if item["source_id"] == "Metadata Only")
    assert metadata_only["chunk_count"] == 0
    assert data["source_count"] == 3
    assert data["chunked_source_count"] == 2
    assert data["source_tier_counts"]["Core EMERGE deliverable"] == 1
    assert data["source_tier_counts"]["Adjacent literature"] == 1


def test_sources_csv_endpoint(monkeypatch, tmp_path):
    app_module = import_test_app(monkeypatch, tmp_path)
    client = app_module.app.test_client()

    response = client.get("/sources.csv")

    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    body = response.data.decode("utf-8")
    assert "source_id,citation,title" in body
    assert "EMERGE Test Source" in body


def test_homepage_exposes_required_toolkit_surfaces(monkeypatch, tmp_path):
    app_module = import_test_app(monkeypatch, tmp_path)
    client = app_module.app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    body = response.data.decode("utf-8")
    assert "EMERGE Ethics Toolkit" in body
    assert "Ethics Wiki" in body
    assert "Bibliography Register" in body
    assert "Methodology & Limits" in body
    assert "Stakeholder Checklist" in body
    assert "Ethics Bot" in body
    assert ">Focused</button>" in body
    assert ">Accurate</button>" not in body
    assert "Disclaimer, Privacy & Terms" in body
    assert "prompt-guide" in body
    assert "status-strip" in body
    assert "<strong>Definition.</strong>" in body
    assert "<strong>Assessment focus.</strong>" in body
    assert "<strong>Governance implication.</strong>" in body
    assert "Connects to:" in body
    assert "capacities, tasks, environments and metrics" in body
    assert "EMERGE D1.1; EMERGE D1.2; EMERGE D1.3" in body
    assert "rather than searched for retrospectively as a single culprit" in body
    assert "Evidence:" in body
    assert "Risk if absent:" in body
    assert "Are responsibility routes defined before deployment?" in body
    assert "Can incidents be traced to inputs, components, versions, and human decisions?" in body
    assert "Is ethics treated as part of the research lifecycle rather than a final paragraph?" in body
    assert "Is there a non-AI alternative or human review route for important decisions?" in body
    assert "Regulation (EU) 2024/1689" in body
    assert "sourceLookup('D1.1')" in body
    assert "sourceLookup('2024/1689')" in body
    assert 'class="src-ref"' in body
    assert "source-filters" in body
    assert "setSourceTier('__ocr')" in body
    assert "setSourceTier('', false)" in body
    assert "No matching sources in this filter" in body
    assert "Clear filter" in body
    assert "method-status" in body
    assert "loadMethodStatus" in body
    assert "Reset checks" in body
    assert "renderSourceButtons" in body
    assert "emergeChecklist:" in body
    assert "Chats may be logged" in body


def test_vignette_feature_is_not_exposed():
    body = read_static_index().lower()

    assert "vignettes" not in body
    assert "generate question" not in body
    assert "/vignettes" not in body
    assert not (ROOT / "vignettes.json").exists()


def test_evaluation_endpoint_exposes_seed_metrics(monkeypatch, tmp_path):
    app_module = import_test_app(monkeypatch, tmp_path)
    client = app_module.app.test_client()

    response = client.get("/evaluation")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "seed"
    assert data["question_count"] > 0
    assert "open_ended" in data["question_types"]
    assert "multiple_choice" in data["question_types"]
    assert "citation_precision" in data["metrics"]["open_ended"]
    assert all("vignette" not in item["question"].lower() for item in data["questions"])


def test_empty_chat_request_returns_400(monkeypatch, tmp_path):
    app_module = import_test_app(monkeypatch, tmp_path)
    client = app_module.app.test_client()

    response = client.post("/chat", json={"messages": []})

    assert response.status_code == 400
    assert response.get_json()["error"] == "No messages provided"


def test_greeting_triage_does_not_require_api_key(monkeypatch, tmp_path):
    app_module = import_test_app(monkeypatch, tmp_path)
    client = app_module.app.test_client()

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["scope"] == "needs_clarification"
    assert data["classification"] == "greeting"
    assert data["chunks_used"] == 0


def test_admin_disabled_without_password(monkeypatch, tmp_path):
    app_module = import_test_app(monkeypatch, tmp_path)
    client = app_module.app.test_client()

    response = client.get("/admin/chats")

    assert response.status_code == 503
    assert b"Admin chat logs are disabled" in response.data


def test_admin_basic_auth_with_password(monkeypatch, tmp_path):
    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text(
        json.dumps(
            [
                {
                    "text": "[Source: EMERGE Test]\nTrust and transparency matter for AI ethics.",
                    "source": "EMERGE Test",
                    "chunk_id": 0,
                }
            ]
        ),
        encoding="utf-8",
    )
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    source_metadata_file = tmp_path / "source_metadata.json"
    source_metadata_file.write_text(json.dumps({"schema_version": 1, "sources": []}), encoding="utf-8")
    monkeypatch.setenv("CHUNKS_FILE", str(chunks_file))
    monkeypatch.setenv("DOCUMENTS_DIR", str(documents_dir))
    monkeypatch.setenv("VECTOR_INDEX_FILE", str(tmp_path / "vector_index.json.gz"))
    monkeypatch.setenv("SOURCE_METADATA_FILE", str(source_metadata_file))
    monkeypatch.setenv("EVALUATION_FILE", str(ROOT / "evaluation_questions.json"))
    monkeypatch.setenv("CHAT_LOG_DB", str(tmp_path / "chat_logs.db"))
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "test-secret")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")
    client = app_module.app.test_client()
    token = base64.b64encode(b"admin:secret").decode("ascii")

    response = client.get("/admin/chats", headers={"Authorization": f"Basic {token}"})

    assert response.status_code == 200
    assert b"EMERGE Chat Logs" in response.data
