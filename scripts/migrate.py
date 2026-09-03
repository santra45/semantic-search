"""Apply the numbered SQL migrations in migrations/, once each, in order.

    python -m scripts.migrate              # dry run: says what would run
    python -m scripts.migrate --apply      # runs it
    python -m scripts.migrate --status     # what is applied, what is pending

Dry run is the default, deliberately and for the same reason
scripts/migrate_v2_schema.py made that choice: the command that shows you the
plan and the command that changes the database should not be the same
keystrokes.

WHY NOT ALEMBIC. The codebase is raw `text()` SQL end to end and the v2 schema
lives in schema_v2.py as plain DDL strings. A migration tool that wants models
to diff against would need an ORM layer that exists nowhere else, so the whole
apparatus would be carried for one directory. This is ~200 lines and does the
four things that actually matter: run each file once, run them in order, refuse
to run one that changed since it ran, and leave a record.

────────────────────────────────────────────────────────────────────────────
MYSQL DDL DOES NOT ROLL BACK.

CREATE TABLE and ALTER TABLE commit implicitly. There is no transaction around
a migration and this runner does not pretend otherwise: a file that fails on
its fourth statement leaves the first three applied, and is recorded as NOT
applied so the next run retries it from the top.

That is why every migration in this directory MUST be idempotent statement by
statement — guarded with an information_schema check, or written so re-running
is a no-op. A migration that is only idempotent as a whole is a migration that
cannot be resumed after it half-fails at 3am.
────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import hashlib
import re
import sys
import time
from pathlib import Path

# Import for the side effect of load_dotenv() as much as for the engine: run
# from a shell without the container's environment and DB_URL is built from
# empty strings, which fails as an auth error rather than as "no config".
from backend.app.services.database import engine

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

# NNN_name.sql — the number is the version and it is what ordering and the
# applied-set are keyed on, so a file may be renamed but never renumbered.
_FILENAME_RE = re.compile(r"^(\d{3})_([a-z0-9_]+)\.sql$")

SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS `schema_migrations` (
  `version`     CHAR(3)      CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
  `filename`    VARCHAR(255) NOT NULL,
  `checksum`    CHAR(64)     CHARACTER SET ascii COLLATE ascii_bin NOT NULL
                COMMENT 'SHA-256 of the file as applied. A mismatch means the file was edited after running.',
  `applied_at`  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `duration_ms` INT          NOT NULL DEFAULT 0,
  PRIMARY KEY (`version`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
"""


# ── Statement splitting ──────────────────────────────────────────────────────

def split_statements(sql: str) -> list[str]:
    """Split a migration file into statements on unquoted, uncommented `;`.

    Hand-rolled because the alternatives are worse. pymysql's MULTI_STATEMENTS
    flag would let the server split it, but it has to be set when the connection
    is created and this runner borrows the application's engine. A naive
    sql.split(";") breaks the moment a COMMENT or a string literal contains a
    semicolon — and migration 001's column comments are exactly the kind of
    prose that eventually will.

    Handles: '' and "" strings (with backslash and doubled-quote escapes),
    `backtick` identifiers, `--` and `#` line comments, and /* */ blocks.
    Deliberately does NOT handle DELIMITER, so no triggers or stored procedures
    in this directory. If one is ever needed, that is the day to reach for the
    mysql client instead of extending this.
    """
    statements: list[str] = []
    buf: list[str] = []
    i, n = 0, len(sql)
    quote: str | None = None          # the char that closes the current literal

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        if quote:
            buf.append(ch)
            if ch == "\\" and quote in "'\"":
                # Backslash escape: consume the next char whole so \' does not
                # read as a close quote.
                if i + 1 < n:
                    buf.append(nxt)
                    i += 2
                    continue
            elif ch == quote:
                # Doubled quote ('') is an escaped quote, not a close.
                if nxt == quote:
                    buf.append(nxt)
                    i += 2
                    continue
                quote = None
            i += 1
            continue

        # -- comment (SQL requires the trailing space/newline; -->, --- are fine)
        if ch == "-" and nxt == "-":
            j = sql.find("\n", i)
            i = n if j == -1 else j + 1
            buf.append("\n")
            continue

        if ch == "#":
            j = sql.find("\n", i)
            i = n if j == -1 else j + 1
            buf.append("\n")
            continue

        if ch == "/" and nxt == "*":
            j = sql.find("*/", i + 2)
            i = n if j == -1 else j + 2
            buf.append(" ")
            continue

        if ch in "'\"`":
            quote = ch
            buf.append(ch)
            i += 1
            continue

        if ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


# ── Discovery ────────────────────────────────────────────────────────────────

def discover() -> list[tuple[str, Path, str]]:
    """(version, path, checksum) for every migration, ordered by version."""
    if not MIGRATIONS_DIR.is_dir():
        sys.exit(f"No migrations directory at {MIGRATIONS_DIR}")

    found = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        m = _FILENAME_RE.match(path.name)
        if not m:
            sys.exit(
                f"Bad migration filename: {path.name}\n"
                f"Expected NNN_lowercase_with_underscores.sql (e.g. 004_alert_rules.sql).\n"
                f"Refusing to run rather than skipping it silently — a migration "
                f"the runner cannot see is a schema change that never ships."
            )
        found.append((m.group(1), path, hashlib.sha256(path.read_bytes()).hexdigest()))

    versions = [v for v, _, _ in found]
    dupes = {v for v in versions if versions.count(v) > 1}
    if dupes:
        sys.exit(f"Duplicate migration version(s): {', '.join(sorted(dupes))}")
    return found


def table_exists(conn, name: str) -> bool:
    return bool(conn.exec_driver_sql(
        "SELECT COUNT(*) FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s",
        (name,),
    ).scalar())


def applied_map(conn) -> dict[str, dict]:
    """What has run. Empty dict when the bookkeeping table does not exist yet.

    Checked rather than created, because this is called on the dry-run path too
    and a dry run must write NOTHING — including the table it would like to read.
    A command whose whole promise is "this changes nothing" cannot leave a
    CREATE TABLE behind, however harmless that particular table is.
    """
    if not table_exists(conn, "schema_migrations"):
        return {}
    rows = conn.exec_driver_sql(
        "SELECT version, filename, checksum, applied_at FROM schema_migrations"
    ).mappings().all()
    return {r["version"]: dict(r) for r in rows}


# ── Running ──────────────────────────────────────────────────────────────────

def run_one(conn, version: str, path: Path, checksum: str) -> None:
    """Execute every statement in one file, on ONE connection.

    The single connection is not incidental. Migration 001 sets a user variable
    in one statement and reads it in the next, and @vars are per-session — run
    the statements on a pool that hands out a different connection and the
    guard silently reads NULL, which evaluates as "column does not exist" and
    quietly does the wrong thing rather than failing.
    """
    statements = split_statements(path.read_text(encoding="utf-8"))
    started = time.time()

    for n, stmt in enumerate(statements, 1):
        try:
            # exec_driver_sql, NOT text(): SQLAlchemy reads `:name` in a text()
            # as a bind parameter, and migration SQL is full of `:=`. Straight to
            # the DBAPI means the file runs as written.
            result = conn.exec_driver_sql(stmt)
        except Exception as exc:
            raise RuntimeError(
                f"{path.name} failed at statement {n}/{len(statements)}:\n"
                f"  {stmt.strip()[:300]}\n"
                f"  {type(exc).__name__}: {exc}\n"
                f"  NOT recorded as applied. MySQL does not roll back DDL, so "
                f"statements 1..{n - 1} are still in place — fix the file and "
                f"re-run; every migration here is written to be re-runnable."
            ) from exc

        # Migrations carry diagnostic SELECTs (001 prints what it is about to
        # drop). Showing them is the whole reason they are in there.
        if result.returns_rows:
            rows = result.mappings().all()
            for row in rows:
                print("        | " + "  ".join(f"{k}={v}" for k, v in row.items()))

    duration_ms = int((time.time() - started) * 1000)
    conn.exec_driver_sql(
        "INSERT INTO schema_migrations (version, filename, checksum, duration_ms) "
        "VALUES (%s, %s, %s, %s)",
        (version, path.name, checksum, duration_ms),
    )
    print(f"    applied in {duration_ms} ms ({len(statements)} statements)")


def main(argv: list[str]) -> int:
    apply = "--apply" in argv
    status_only = "--status" in argv

    migrations = discover()

    with engine.connect() as conn:
        # Created only on the write path. See applied_map().
        if apply:
            conn.exec_driver_sql(SCHEMA_MIGRATIONS_DDL)
            try:
                conn.commit()
            except Exception:
                pass

        done = applied_map(conn)
        if not done and not table_exists(conn, "schema_migrations"):
            print("note: schema_migrations does not exist yet — treating every "
                  "migration as pending. It is created on the first --apply.\n")

        # A file that changed after it ran is the failure this table exists to
        # catch: the database no longer matches the migration that claims to
        # describe it, and nothing downstream will notice on its own.
        drifted = [
            (v, p.name, done[v]["checksum"], c)
            for v, p, c in migrations
            if v in done and done[v]["checksum"] != c
        ]
        if drifted:
            print("CHECKSUM MISMATCH — these ran, then their files were edited:\n")
            for v, name, was, now in drifted:
                print(f"  {v}  {name}\n      applied {was[:16]}…  on disk {now[:16]}…")
            print(
                "\nThe database reflects what ran, not what the file says now. Add a "
                "NEW migration with the change; do not edit one that has run.\n"
                "(If the edit was cosmetic and you are certain, update the checksum "
                "in schema_migrations by hand — deliberately fiddly.)"
            )
            return 2

        pending = [(v, p, c) for v, p, c in migrations if v not in done]

        print(f"migrations dir : {MIGRATIONS_DIR}")
        print(f"applied        : {len(done)}")
        print(f"pending        : {len(pending)}\n")

        for v, p, _ in migrations:
            if v in done:
                print(f"  [x] {v}  {p.name:<48} {done[v]['applied_at']}")
            else:
                print(f"  [ ] {v}  {p.name}")
        print()

        if status_only:
            return 0

        if not pending:
            print("Nothing to do.")
            return 0

        if not apply:
            print("DRY RUN — nothing was written. Re-run with --apply to execute:")
            for v, p, _ in pending:
                print(f"    {v}  {p.name}")
            return 0

        for v, p, c in pending:
            print(f"--> {v}  {p.name}")
            run_one(conn, v, p, c)
            try:
                conn.commit()
            except Exception:
                pass

        print(f"\nDone. {len(pending)} migration(s) applied.")
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
