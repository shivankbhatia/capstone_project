import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ABLATION_LOCK_SHA256 = "840c12db474735757ece08ae70b489795870c09933dfd18819a34a32de895095"
EXPECTED_CLASSIFIER_LOCK_SHA256 = "46e7c2008f89c772158e3f11aedb4af799e09b1fbc30a0c628df6c079731509a"
EXPECTED_MANIFEST_SHA256 = "de2fd72215e6f2766cc66c629ebd08fd9a1c55b6dab40c63a2e7890a0a8bac7b"


def test_ablation_lock_is_immutable_and_parent_hashes_match():
    lock_path = ROOT / "splits/ablation_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    classifier_hash = hashlib.sha256(
        (ROOT / "splits/finalist_lock.json").read_bytes()
    ).hexdigest()
    manifest_hash = hashlib.sha256(
        (ROOT / "splits/heldout_manifest.json").read_bytes()
    ).hexdigest()

    assert lock_hash == EXPECTED_ABLATION_LOCK_SHA256
    assert classifier_hash == EXPECTED_CLASSIFIER_LOCK_SHA256
    assert manifest_hash == EXPECTED_MANIFEST_SHA256
    assert lock["parent_locks"]["classifier_lock_sha256"] == classifier_hash
    assert lock["parent_locks"]["manifest_sha256"] == manifest_hash
