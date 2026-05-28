import base64
import importlib
import json
import sys


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
    monkeypatch.setenv("CHUNKS_FILE", str(chunks_file))
    monkeypatch.setenv("DOCUMENTS_DIR", str(documents_dir))
    monkeypatch.setenv("VECTOR_INDEX_FILE", str(tmp_path / "vector_index.json.gz"))
    monkeypatch.setenv("CHAT_LOG_DB", str(tmp_path / "chat_logs.db"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("ADMIN_SESSION_SECRET", raising=False)
    sys.modules.pop("app", None)
    return importlib.import_module("app")


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
    monkeypatch.setenv("CHUNKS_FILE", str(chunks_file))
    monkeypatch.setenv("DOCUMENTS_DIR", str(documents_dir))
    monkeypatch.setenv("VECTOR_INDEX_FILE", str(tmp_path / "vector_index.json.gz"))
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
