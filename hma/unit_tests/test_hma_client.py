"""Tests for hma.hma_client, mocking the HMA HTTP layer.

These never make real network calls. The live path (a running HMA service) is
exercised by the smoke test and the evaluation suites.
"""

from unittest.mock import MagicMock, patch

import pytest
from requests import RequestException

from hma.hma_client import HMAClient


def _response(ok=True, status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = status_code
    resp.text = text
    if isinstance(json_data, Exception):
        resp.json.side_effect = json_data
    else:
        resp.json.return_value = json_data
    return resp


@pytest.fixture
def client():
    return HMAClient()


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "a.jpg"
    path.write_bytes(b"not-really-a-jpeg")
    return str(path)


@patch("hma.hma_client.requests")
class TestBanks:
    def test_bank_exists_true(self, requests_mock, client):
        requests_mock.get.return_value = _response(ok=True)
        assert client.bank_exists("B") is True

    def test_bank_exists_false_on_404(self, requests_mock, client):
        requests_mock.get.return_value = _response(ok=False, status_code=404)
        assert client.bank_exists("B") is False

    def test_bank_exists_false_on_server_error(self, requests_mock, client):
        requests_mock.get.return_value = _response(ok=False, status_code=500, text="boom")
        assert client.bank_exists("B") is False

    def test_bank_exists_false_on_request_exception(self, requests_mock, client):
        requests_mock.get.side_effect = RequestException("down")
        assert client.bank_exists("B") is False

    def test_create_bank_posts_name(self, requests_mock, client):
        requests_mock.post.return_value = _response(ok=True)
        assert client.create_bank("B") is True
        assert requests_mock.post.call_args.kwargs["json"] == {
            "name": "B",
            "matching_enabled_ratio": 1.0,
        }

    def test_create_bank_failure(self, requests_mock, client):
        requests_mock.post.return_value = _response(ok=False, status_code=403, text="exists")
        assert client.create_bank("B") is False

    def test_setup_bank_creates_when_missing(self, requests_mock, client):
        requests_mock.get.return_value = _response(ok=False, status_code=404)
        requests_mock.post.return_value = _response(ok=True)
        assert client.setup_bank("B") is True

    def test_setup_bank_fails_when_create_fails(self, requests_mock, client):
        requests_mock.get.return_value = _response(ok=False, status_code=404)
        requests_mock.post.return_value = _response(ok=False, status_code=500, text="boom")
        assert client.setup_bank("B") is False

    def test_delete_bank(self, requests_mock, client):
        requests_mock.delete.return_value = _response(ok=True)
        assert client.delete_bank("B") is True
        requests_mock.delete.return_value = _response(ok=False, status_code=404, text="nope")
        assert client.delete_bank("B") is False


@patch("hma.hma_client.requests")
class TestAddAndUpload:
    def test_add_file_extracts_content_id_as_string(self, requests_mock, client, image):
        requests_mock.post.return_value = _response(json_data={"id": 7, "signals": {}}, text="{}")
        result = client.add_file_to_hma_bank(image, "B")
        assert result["status"] == "success"
        assert result["content_id"] == "7"

    def test_add_file_without_json_body_has_no_content_id(self, requests_mock, client, image):
        import json

        requests_mock.post.return_value = _response(
            json_data=json.JSONDecodeError("bad", "", 0), text="<html>"
        )
        result = client.add_file_to_hma_bank(image, "B")
        assert result["status"] == "success"
        assert result["content_id"] is None

    def test_add_file_failure(self, requests_mock, client, image):
        requests_mock.post.return_value = _response(ok=False, status_code=400, text="bad")
        result = client.add_file_to_hma_bank(image, "B")
        assert result["status"] == "failure"

    def test_upload_files_maps_content_ids_and_skips_failures(self, requests_mock, client, image, tmp_path):
        other = tmp_path / "b.jpg"
        other.write_bytes(b"x")
        requests_mock.post.side_effect = [
            _response(json_data={"id": 1}, text="{}"),
            _response(ok=False, status_code=500, text="boom"),
        ]
        assert client.upload_files_to_bank([image, str(other)], "B") == {"1": image}


@patch("hma.hma_client.requests")
class TestAddSignal:
    def test_posts_the_signal_and_returns_the_content_id_as_a_string(self, requests_mock, client):
        requests_mock.post.return_value = _response(json_data={"id": 12, "signals": {"clip_float": "ab"}})
        assert client.add_signal_to_bank("B", "clip_float", "ab") == "12"
        call = requests_mock.post.call_args
        assert call.args[0].endswith("/c/bank/B/signal")
        assert call.kwargs["json"] == {"clip_float": "ab"}

    def test_failure_returns_none(self, requests_mock, client):
        requests_mock.post.return_value = _response(ok=False, status_code=400, text="bad signal")
        assert client.add_signal_to_bank("B", "clip_float", "zz") is None

    def test_request_exception_returns_none(self, requests_mock, client):
        requests_mock.post.side_effect = RequestException("down")
        assert client.add_signal_to_bank("B", "clip_float", "ab") is None

    def test_a_response_without_an_id_returns_none(self, requests_mock, client):
        requests_mock.post.return_value = _response(json_data={})
        assert client.add_signal_to_bank("B", "clip_float", "ab") is None


@patch("hma.hma_client.requests")
class TestHashImageBytes:
    def test_returns_one_signal_per_image_in_order(self, requests_mock, client):
        requests_mock.post.return_value = _response(json_data=[{"clip_float": "1"}, {"clip_float": "2"}])
        assert client.hash_image_bytes([b"a", b"b"], "clip_float") == ["1", "2"]
        call = requests_mock.post.call_args
        assert [name for name, _ in call.kwargs["files"]] == ["photo", "photo"]
        assert call.kwargs["params"] == {"signal_type": "clip_float"}

    @pytest.mark.parametrize(
        "response",
        [
            _response(ok=False, status_code=500, text="boom"),
            _response(json_data=[{"clip_float": "1"}]),
            _response(json_data={"clip_float": "1"}),
            _response(json_data=[{"clip_float": "1"}, {"error": "bad image"}]),
        ],
    )
    def test_any_failure_is_an_error(self, requests_mock, client, response):
        requests_mock.post.return_value = response
        with pytest.raises(RuntimeError):
            client.hash_image_bytes([b"a", b"b"], "clip_float")

    def test_request_exception_is_an_error(self, requests_mock, client):
        requests_mock.post.side_effect = RequestException("down")
        with pytest.raises(RuntimeError, match="down"):
            client.hash_image_bytes([b"a"], "clip_float")


@patch("hma.hma_client.requests")
class TestHashing:
    def test_hash_local_content_returns_json(self, requests_mock, client, image):
        requests_mock.post.return_value = _response(json_data={"clip_float": "abcd"})
        assert client.hash_local_content(image) == {"clip_float": "abcd"}

    def test_hash_local_content_reports_http_error(self, requests_mock, client, image):
        requests_mock.post.return_value = _response(ok=False, status_code=500, text="boom")
        result = client.hash_local_content(image)
        assert result["success"] is False
        assert result["status_code"] == 500

    def test_batch_returns_results_in_input_order(self, requests_mock, client, image, tmp_path):
        other = tmp_path / "b.jpg"
        other.write_bytes(b"x")
        requests_mock.post.return_value = _response(
            json_data=[{"clip_float": "1"}, {"clip_float": "2"}]
        )
        assert client.hash_local_content_batch([image, str(other)], signal_type="clip_float") == [
            {"clip_float": "1"},
            {"clip_float": "2"},
        ]
        assert requests_mock.post.call_args.kwargs["params"] == {"signal_type": "clip_float"}

    def test_batch_splits_into_batches(self, requests_mock, client, image):
        requests_mock.post.return_value = _response(json_data=[{"clip_float": "1"}])
        result = client.hash_local_content_batch([image, image, image], batch_size=1)
        assert len(result) == 3
        assert requests_mock.post.call_count == 3

    def test_batch_falls_back_on_unexpected_format(self, requests_mock, client, image):
        requests_mock.post.return_value = _response(json_data={"not": "a list"})
        with patch.object(client, "hash_local_content", return_value={"clip_float": "x"}) as single:
            assert client.hash_local_content_batch([image]) == [{"clip_float": "x"}]
        single.assert_called_once_with(image)

    def test_batch_falls_back_on_http_error(self, requests_mock, client, image):
        requests_mock.post.return_value = _response(ok=False, status_code=500, text="boom")
        with patch.object(client, "hash_local_content", return_value={"clip_float": "x"}):
            assert client.hash_local_content_batch([image]) == [{"clip_float": "x"}]

    def test_batch_falls_back_on_request_exception(self, requests_mock, client, image):
        requests_mock.post.side_effect = RequestException("down")
        with patch.object(client, "hash_local_content", return_value={"clip_float": "x"}):
            assert client.hash_local_content_batch([image]) == [{"clip_float": "x"}]


@patch("hma.hma_client.requests")
class TestMatching:
    def test_topk_success(self, requests_mock, client):
        requests_mock.post.return_value = _response(
            json_data={"matches": [{"bank_content_id": 1, "distance": 0.1}]}
        )
        result = client.match_with_signal_topk("sig", 3, "clip_float")
        assert result["status"] == "success"
        assert result["matches"] == [{"bank_content_id": 1, "distance": 0.1}]
        assert requests_mock.post.call_args.kwargs["json"] == {
            "signal_type": "clip_float",
            "signal": "sig",
            "k": 3,
        }

    def test_threshold_success(self, requests_mock, client):
        requests_mock.post.return_value = _response(json_data={"matches": []})
        result = client.match_with_signal_threshold("sig", 0.2, "clip_float")
        assert result["status"] == "success"
        assert requests_mock.post.call_args.kwargs["json"]["threshold"] == 0.2

    def test_lookup_failure_and_exception(self, requests_mock, client):
        requests_mock.post.return_value = _response(ok=False, status_code=500, text="boom")
        assert client.match_with_signal_topk("sig", 1, "clip_float")["status"] == "failure"
        requests_mock.post.side_effect = RequestException("down")
        assert client.match_with_signal_threshold("sig", 0.1, "clip_float")["status"] == "failure"

    def test_match_local_content_topk_fails_without_hash(self, requests_mock, client, image):
        with patch.object(client, "hash_local_content", return_value={"pdq": "x"}):
            result = client.match_local_content_topk(image, 2, "clip_float")
        assert result["status"] == "failure"
        requests_mock.post.assert_not_called()

    def test_compare_hashes(self, requests_mock, client):
        requests_mock.post.return_value = _response(json_data={"clip_float": [True, {"distance": "0.05"}]})
        assert client.compare_hashes("a", "b", "clip_float") == {
            "status": "success",
            "result": [True, {"distance": "0.05"}],
        }
        requests_mock.post.return_value = _response(ok=False, status_code=400, text="bad")
        assert client.compare_hashes("a", "b", "clip_float")["status"] == "failure"


@patch("hma.hma_client.time.sleep")
@patch("hma.hma_client.requests")
class TestIndex:
    def test_get_index_size(self, requests_mock, sleep_mock, client):
        requests_mock.get.return_value = _response(json_data={"clip_float": {"size": 12}})
        assert client.get_index_size("clip_float") == 12

    def test_get_index_size_defaults_to_zero(self, requests_mock, sleep_mock, client):
        requests_mock.get.return_value = _response(json_data={})
        assert client.get_index_size("clip_float") == 0

    def test_wait_returns_once_expected_size_reached(self, requests_mock, sleep_mock, client):
        requests_mock.get.side_effect = [
            _response(json_data={"clip_float": {"size": 1}}),
            _response(json_data={"clip_float": {"size": 5}}),
        ]
        client.wait_for_index_update(expected_size=5, signal_type="clip_float", max_wait=60)
        assert sleep_mock.call_count == 1

    def test_wait_gives_up_after_max_wait(self, requests_mock, sleep_mock, client):
        requests_mock.get.return_value = _response(json_data={"clip_float": {"size": 0}})
        client.wait_for_index_update(expected_size=5, signal_type="clip_float", max_wait=15)
        assert requests_mock.get.call_count == 3


class TestTestEnvironmentReset:
    def test_create_fresh_database_needs_psycopg2(self, client):
        with patch("hma.hma_client.PSYCOPG2_AVAILABLE", False):
            assert client.create_fresh_database() is False

    def test_cleanup_fails_when_database_clear_fails(self, client):
        with patch.object(client, "create_fresh_database", return_value=False), patch.object(
            client, "get_index_size"
        ) as index_size:
            assert client.cleanup_test_environment("clip_float") is False
        index_size.assert_not_called()

    def test_cleanup_checks_the_given_signal_type_index(self, client):
        with patch.object(client, "create_fresh_database", return_value=True), patch(
            "hma.hma_client.time.sleep"
        ), patch.object(client, "get_index_size", return_value=0) as index_size:
            assert client.cleanup_test_environment("pdq") is True
        index_size.assert_called_once_with("pdq")

    def test_cleanup_defaults_signal_type_from_env(self, client, monkeypatch):
        monkeypatch.setenv("SIGNAL_TYPE", "clip")
        with patch.object(client, "create_fresh_database", return_value=True), patch(
            "hma.hma_client.time.sleep"
        ), patch.object(client, "get_index_size", return_value=3) as index_size:
            assert client.cleanup_test_environment() is True  # a non-empty index only warns
        index_size.assert_called_once_with("clip")
