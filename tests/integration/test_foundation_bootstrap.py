from pathlib import Path

from chatbox.app.bootstrap import AppContainer, create_app_container
from chatbox.storage.sqlite_metadata import SqliteMetadataStore


def test_bootstrap_builds_container(tmp_path: Path) -> None:
    db_path = tmp_path / "chatbox.db"
    container = create_app_container(db_path)

    assert isinstance(container, AppContainer)
    assert isinstance(container.metadata_store, SqliteMetadataStore)
    assert db_path.exists()
