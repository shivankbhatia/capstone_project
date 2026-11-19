import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import prototype.server as prototype_server


@pytest.fixture
def prototype_api(monkeypatch):
    monkeypatch.setattr(prototype_server, "SERVICE", prototype_server.ReplayService())
    server = ThreadingHTTPServer(("127.0.0.1", 0), prototype_server.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join()
    server.server_close()


def request_json(base_url, path, body=None, content_type="application/json"):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(f"{base_url}{path}", data=data, headers={"Content-Type": content_type})
    try:
        with urlopen(request) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def test_health_contract_exposes_locked_arm_metadata(prototype_api):
    status, payload = request_json(prototype_api, "/api/health")

    assert status == 200
    assert payload["supported_arms"] == ["eeg_only", "fixed_lm"]
    assert payload["fixed_lm"]["alpha"] == 0.85
    assert payload["locked_special_key_map"]["matches_lock"] is True


def test_simulation_contract_labels_aac_as_simulated(prototype_api):
    status, payload = request_json(
        prototype_api, "/api/simulate", {"message": "ignored", "arm": "eeg_only", "scenario": "aac_simulation"}
    )

    assert status == 200
    assert payload["source"] == "custom_simulation"
    assert payload["scenario"] == "aac_simulation"
    assert payload["target"] == "I NEED WATER"
    assert payload["selections"][0]["trace"][0]["flash_mode"] == "simulated_order"


@pytest.mark.parametrize(
    ("body", "content_type", "expected_status"),
    [
        ({"message": "A"}, "text/plain", 415),
        ({"message": "A", "scan_mode": "compact"}, "application/json", 400),
        ({"message": "A" * (17 * 1024)}, "application/json", 413),
    ],
)
def test_post_contract_rejects_invalid_requests(prototype_api, body, content_type, expected_status):
    status, payload = request_json(prototype_api, "/api/simulate", body, content_type)

    assert status == expected_status
    assert "error" in payload
