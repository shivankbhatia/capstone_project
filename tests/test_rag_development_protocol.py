import csv

from scripts.run_rag_development_benchmark import write_bank_from_training_text
from scripts.prepare_rag_development_corpora import build


def test_bank_builder_never_receives_or_serializes_evaluation_text(tmp_path):
    train_text = "TRAIN_ONLY_MARKER alpha beta."
    evaluation_text = "EVALUATION_SECRET_MARKER unseen phrase."
    bank = tmp_path / "bank.csv"

    # The bank API accepts only the training split. Evaluation text is not an
    # argument, so it cannot be accidentally mixed into this offline bank.
    write_bank_from_training_text(bank, train_text)
    with bank.open(newline="", encoding="utf-8") as stream:
        content = " ".join(row["text"] for row in csv.DictReader(stream))
    assert "TRAIN_ONLY_MARKER" in content
    assert "EVALUATION_SECRET_MARKER" not in content
    assert evaluation_text not in content


def test_public_domain_dev_split_is_author_separated_and_chronological(tmp_path):
    manifest = build(max_words=500, train_fraction=0.6, tune_fraction=0.2,
                     output_path=tmp_path / "split.json")
    assert manifest["development_only"] is True
    assert len(manifest["authors"]) == 4
    for author in manifest["authors"].values():
        assert author["train_words"] >= 300
        assert author["tune_words"] >= 100
        assert author["evaluation_words"] >= 100
