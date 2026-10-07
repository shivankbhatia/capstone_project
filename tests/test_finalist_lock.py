import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LOCK_SHA256 = "46e7c2008f89c772158e3f11aedb4af799e09b1fbc30a0c628df6c079731509a"
EXPECTED_MANIFEST_SHA256 = "de2fd72215e6f2766cc66c629ebd08fd9a1c55b6dab40c63a2e7890a0a8bac7b"


def test_finalist_lock_is_immutable_and_manifest_matches():
    lock_path = ROOT / "splits/finalist_lock.json"
    manifest_path = ROOT / "splits/heldout_manifest.json"
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    lock_hash = hashlib.sha256(lock_bytes).hexdigest()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    assert lock_hash == EXPECTED_LOCK_SHA256
    assert lock["manifest_sha256"] == EXPECTED_MANIFEST_SHA256
    assert manifest_hash == EXPECTED_MANIFEST_SHA256
