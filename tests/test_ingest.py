from __future__ import annotations

from pathlib import Path

from conftest import CountingEmbeddings, with_settings
from docs_assistant.config import Settings
from docs_assistant.services import build_ingestor


def test_first_run_indexes_everything(settings: Settings, embeddings: CountingEmbeddings) -> None:
    ingestor = build_ingestor(settings, embedding_provider=embeddings)
    report = ingestor.run()
    assert sorted(report.added) == ["fountain-care.txt", "returns.md", "shipping.md"]
    assert report.chunks_written == 6
    assert report.embeddings_computed == 6
    assert report.embeddings_from_cache == 0
    assert ingestor.store.stats() == {"documents": 3, "chunks": 6}


def test_second_run_is_a_no_op(settings: Settings, embeddings: CountingEmbeddings) -> None:
    ingestor = build_ingestor(settings, embedding_provider=embeddings)
    ingestor.run()
    generation = ingestor.store.generation()
    calls_before = len(embeddings.calls)

    report = ingestor.run()

    assert sorted(report.unchanged) == ["fountain-care.txt", "returns.md", "shipping.md"]
    assert not report.changed
    assert report.chunks_written == 0
    assert len(embeddings.calls) == calls_before  # provider not called at all
    assert ingestor.store.generation() == generation


def test_only_changed_file_is_reprocessed_and_unchanged_chunks_hit_cache(
    settings: Settings, embeddings: CountingEmbeddings, docs_dir: Path
) -> None:
    ingestor = build_ingestor(settings, embedding_provider=embeddings)
    ingestor.run()
    path = docs_dir / "returns.md"
    path.write_text(path.read_text().replace("5 business days", "7 business days"))
    embedded_before = embeddings.texts_embedded

    report = ingestor.run()

    assert report.updated == ["returns.md"]
    assert sorted(report.unchanged) == ["fountain-care.txt", "shipping.md"]
    assert report.embeddings_computed == 1  # only the edited section
    assert report.embeddings_from_cache == 1  # the untouched "Return window" section
    assert embeddings.texts_embedded - embedded_before == 1


def test_deleted_file_is_removed(
    settings: Settings, embeddings: CountingEmbeddings, docs_dir: Path
) -> None:
    ingestor = build_ingestor(settings, embedding_provider=embeddings)
    ingestor.run()
    (docs_dir / "shipping.md").unlink()

    report = ingestor.run()

    assert report.removed == ["shipping.md"]
    assert {d["path"] for d in ingestor.store.list_documents()} == {
        "returns.md",
        "fountain-care.txt",
    }
    assert ingestor.store.stats()["chunks"] == 3


def test_rebuild_reprocesses_everything_without_new_embeddings(
    settings: Settings, embeddings: CountingEmbeddings
) -> None:
    ingestor = build_ingestor(settings, embedding_provider=embeddings)
    ingestor.run()
    embedded_before = embeddings.texts_embedded

    report = ingestor.run(rebuild=True)

    assert len(report.updated) == 3
    assert report.embeddings_computed == 0
    assert report.embeddings_from_cache == 6
    assert embeddings.texts_embedded == embedded_before


def test_changing_chunk_settings_invalidates_documents(
    settings: Settings, embeddings: CountingEmbeddings
) -> None:
    build_ingestor(settings, embedding_provider=embeddings).run()
    smaller = with_settings(settings, chunk_max_chars=60, chunk_overlap_chars=20)

    report = build_ingestor(smaller, embedding_provider=embeddings).run()

    assert len(report.updated) == 3
    assert report.chunks_written > 6


def test_empty_and_unsupported_files_are_skipped(
    settings: Settings, embeddings: CountingEmbeddings, docs_dir: Path
) -> None:
    (docs_dir / "empty.md").write_text("   \n\n")
    (docs_dir / "image.png").write_bytes(b"\x89PNG")
    report = build_ingestor(settings, embedding_provider=embeddings).run()
    assert report.skipped == ["empty.md"]  # .png is not even considered
    assert "empty.md" not in report.added


def test_missing_docs_dir_raises(
    settings: Settings, embeddings: CountingEmbeddings, tmp_path: Path
) -> None:
    missing = with_settings(settings, docs_dir=tmp_path / "nope")
    ingestor = build_ingestor(missing, embedding_provider=embeddings)
    try:
        ingestor.run()
    except FileNotFoundError as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError")
