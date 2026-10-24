"""
Migration runner (app/db/migrations/runner.py) and the shipped migrations.

The migrations are run against a session that records the SQL, so every
upgrade / downgrade is at least executed end to end and its key DDL checked.
"""

from types import SimpleNamespace

import pytest

from app.db.migrations import runner


class RecordingSession:
    """Async session double: records SQL and tracks the _migrations table."""

    def __init__(self, applied=()):
        self.applied = set(applied)
        self.statements: list[str] = []
        self.commits = 0
        self.ran: list[str] = []          # written by the fake migrations below

    async def execute(self, statement, params=None):
        sql = " ".join(str(statement).split())
        self.statements.append(sql)
        if sql.startswith(f"SELECT 1 FROM {runner.MIGRATIONS_TABLE}"):
            return SimpleNamespace(fetchone=lambda: (1,) if params["m"] in self.applied else None)
        if sql.startswith(f"INSERT INTO {runner.MIGRATIONS_TABLE}"):
            self.applied.add(params["m"])
        return SimpleNamespace(fetchone=lambda: None)

    async def commit(self):
        self.commits += 1


SHIPPED = runner._discover_migrations()


class TestShippedMigrations:
    def test_all_files_follow_the_naming_pattern(self):
        names = sorted(p.name for p in runner.MIGRATIONS_DIR.glob("*_migration.py"))
        assert [p.name for _, _, p in SHIPPED] == names
        assert [ts for ts, _, _ in SHIPPED] == sorted(ts for ts, _, _ in SHIPPED)

    @pytest.mark.parametrize("path", [p for _, _, p in SHIPPED], ids=lambda p: p.stem)
    async def test_upgrade_and_downgrade_run_and_commit(self, path):
        mod = await runner._load_migration(path)
        for step in ("upgrade", "downgrade"):
            if not hasattr(mod, step):
                assert step == "downgrade"
                continue
            session = RecordingSession()
            await getattr(mod, step)(session)
            assert session.statements, f"{path.stem}.{step} issued no SQL"
            assert session.commits >= 1, f"{path.stem}.{step} never commits"

    @pytest.mark.parametrize("slug, step, expected", [
        ("initial_schema", "upgrade", "CREATE EXTENSION IF NOT EXISTS vector"),
        ("add_embedding_version", "upgrade", "ADD COLUMN IF NOT EXISTS embedding_version"),
        ("tenant_id_rls", "upgrade", "ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY"),
        ("face_pose_embeddings", "upgrade", "ALTER TABLE face_pose_embeddings ENABLE ROW LEVEL SECURITY"),
        ("face_pose_embeddings", "downgrade", "DROP TABLE IF EXISTS face_pose_embeddings"),
    ])
    async def test_key_statements(self, slug, step, expected):
        path = next(p for _, s, p in SHIPPED if s == slug)
        session = RecordingSession()
        await getattr(await runner._load_migration(path), step)(session)
        assert any(expected in sql for sql in session.statements)


@pytest.fixture
def migrations_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "MIGRATIONS_DIR", tmp_path)

    def write(name, body):
        (tmp_path / name).write_text(body, encoding="utf-8")

    return write


def _records(tag):
    return f"async def upgrade(session):\n    session.ran.append({tag!r})\n"


class TestRunner:
    async def test_no_migrations(self, migrations_dir):
        session = RecordingSession()
        assert await runner.run_migrations(session) == {"applied": 0, "skipped": 0, "errors": []}
        assert "CREATE TABLE IF NOT EXISTS _migrations" in session.statements[0]

    async def test_applies_pending_in_order_and_skips_applied(self, migrations_dir):
        migrations_dir("20260102000000_second_migration.py", _records("second"))
        migrations_dir("20260101000000_first_migration.py", _records("first"))
        migrations_dir("20260103000000_third_migration.py", _records("third"))
        migrations_dir("notes_migration.py", _records("ignored"))      # wrong name: skipped
        session = RecordingSession(applied={"20260102000000_second"})

        summary = await runner.run_migrations(session)

        assert summary == {"applied": 2, "skipped": 1, "errors": []}
        assert session.ran == ["first", "third"]
        assert session.applied == {"20260101000000_first", "20260102000000_second", "20260103000000_third"}

    async def test_stops_at_the_first_failure(self, migrations_dir):
        migrations_dir("20260101000000_broken_migration.py",
                       "async def upgrade(session):\n    raise RuntimeError('bad DDL')\n")
        migrations_dir("20260102000000_later_migration.py", _records("later"))
        session = RecordingSession()

        summary = await runner.run_migrations(session)

        assert summary["applied"] == 0
        assert summary["errors"] == [{"migration": "20260101000000_broken", "error": "bad DDL"}]
        assert session.ran == [] and "20260101000000_broken" not in session.applied

    async def test_migration_without_upgrade_is_an_error(self, migrations_dir):
        migrations_dir("20260101000000_empty_migration.py", "async def downgrade(session):\n    pass\n")
        summary = await runner.run_migrations(RecordingSession())
        assert "has no 'upgrade' function" in summary["errors"][0]["error"]
