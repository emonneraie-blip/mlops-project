"""notify_api : promote.py et rollback.py doivent savoir si l'API sert vraiment la nouvelle version."""
import requests

import registry


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


def test_reload_ok(monkeypatch):
    monkeypatch.setattr(registry.requests, "post",
                        lambda *a, **k: FakeResponse(200, {"old_version": "1", "new_version": "2"}))
    assert registry.notify_api() is True


def test_reload_refused_is_not_a_success(monkeypatch, capsys):
    """MLflow injoignable : /v1/reload répond 503 et l'API garde l'ancienne version."""
    monkeypatch.setattr(registry.requests, "post",
                        lambda *a, **k: FakeResponse(503, {"detail": {"error": "reload impossible"}}))
    assert registry.notify_api() is False
    assert "PAS rechargé" in capsys.readouterr().out


def test_api_down(monkeypatch):
    def down(*args, **kwargs):
        raise requests.ConnectionError("API éteinte")
    monkeypatch.setattr(registry.requests, "post", down)
    assert registry.notify_api() is False
