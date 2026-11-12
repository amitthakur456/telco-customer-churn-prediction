
import os
import sqlite3
from pathlib import Path

import psycopg

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

SQLITE_DB = BASE_DIR / "customer_retention_saas.db"

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

TABLES = ("workspaces", "users", "audit_logs")


def validate_configuration():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is missing. Set it in your terminal first."
        )

    if not SQLITE_DB.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {SQLITE_DB}"
        )


# --------------------------------------------------
# READ SQLITE DATA
# --------------------------------------------------

def read_sqlite_data():

    source = sqlite3.connect(SQLITE_DB)
    source.row_factory = sqlite3.Row

    try:
        workspaces = source.execute("""
            SELECT id, name, created_at
            FROM workspaces
            ORDER BY id
        """).fetchall()

        users = source.execute("""
            SELECT id, workspace_id, full_name, email,
                   password_hash, role, is_active, created_at
            FROM users
            ORDER BY id
        """).fetchall()

        logs = source.execute("""
            SELECT id, workspace_id, actor_user_id,
                   action, details, created_at
            FROM audit_logs
            ORDER BY id
        """).fetchall()

        print("\nSQLite records loaded:")
        print("Workspaces:", len(workspaces))
        print("Users:", len(users))
        print("Audit logs:", len(logs))

        return workspaces, users, logs

    finally:
        source.close()


# --------------------------------------------------
# CREATE POSTGRESQL TABLES
# --------------------------------------------------

def create_postgres_tables(pg):

    pg.execute("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id BIGSERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
                DEFAULT CURRENT_TIMESTAMP
        )
    """)

    pg.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id BIGSERIAL PRIMARY KEY,
            workspace_id BIGINT NOT NULL
                REFERENCES workspaces(id),
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'Admin'
                CHECK (role IN ('Admin', 'Analyst', 'Viewer')),
            is_active SMALLINT NOT NULL DEFAULT 1
                CHECK (is_active IN (0, 1)),
            created_at TIMESTAMPTZ NOT NULL
                DEFAULT CURRENT_TIMESTAMP
        )
    """)

    pg.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id BIGSERIAL PRIMARY KEY,
            workspace_id BIGINT NOT NULL
                REFERENCES workspaces(id),
            actor_user_id BIGINT,
            action TEXT NOT NULL,
            details TEXT,
            created_at TIMESTAMPTZ NOT NULL
                DEFAULT CURRENT_TIMESTAMP
        )
    """)


# --------------------------------------------------
# CHECK TARGET TABLES
# --------------------------------------------------

def check_empty_tables(pg):

    for table in TABLES:

        count = pg.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        if count > 0:
            raise RuntimeError(
                f"Migration stopped: {table} already has "
                f"{count} records in PostgreSQL."
            )


# --------------------------------------------------
# INSERT DATA
# --------------------------------------------------

def insert_records(pg, workspaces, users, logs):

    with pg.cursor() as cursor:

        # Insert workspaces
        cursor.executemany(
            """
            INSERT INTO workspaces
                (id, name, created_at)
            VALUES (
                %s, %s,
                COALESCE(%s::timestamptz, CURRENT_TIMESTAMP)
            )
            """,
            [
                (r["id"], r["name"], r["created_at"])
                for r in workspaces
            ]
        )

        print("Workspaces migrated.")

        # Insert users
        cursor.executemany(
            """
            INSERT INTO users
                (id, workspace_id, full_name, email,
                 password_hash, role, is_active, created_at)
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                COALESCE(%s::timestamptz, CURRENT_TIMESTAMP)
            )
            """,
            [
                (
                    r["id"],
                    r["workspace_id"],
                    r["full_name"],
                    r["email"],
                    r["password_hash"],
                    r["role"],
                    r["is_active"],
                    r["created_at"]
                )
                for r in users
            ]
        )

        print("Users migrated.")

        # Insert audit logs
        cursor.executemany(
            """
            INSERT INTO audit_logs
                (id, workspace_id, actor_user_id,
                 action, details, created_at)
            VALUES (
                %s, %s, %s, %s, %s,
                COALESCE(%s::timestamptz, CURRENT_TIMESTAMP)
            )
            """,
            [
                (
                    r["id"],
                    r["workspace_id"],
                    r["actor_user_id"],
                    r["action"],
                    r["details"],
                    r["created_at"]
                )
                for r in logs
            ]
        )

        print("Audit logs migrated.")


# --------------------------------------------------
# RESET AUTO-INCREMENT SEQUENCES
# --------------------------------------------------

def reset_sequences(pg):

    for table in TABLES:

        pg.execute(
            f"""
            SELECT setval(
                pg_get_serial_sequence('{table}', 'id'),
                COALESCE(MAX(id), 1),
                COUNT(*) > 0
            )
            FROM {table}
            """
        )


# --------------------------------------------------
# VERIFY MIGRATION
# --------------------------------------------------

def verify_migration(pg, expected):

    print("\nPostgreSQL verification:")

    for table in TABLES:

        actual = pg.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        expected_count = expected[table]

        print(f"{table}: {actual} / {expected_count}")

        if actual != expected_count:
            raise RuntimeError(
                f"Verification failed for {table}."
            )


# --------------------------------------------------
# MAIN MIGRATION
# --------------------------------------------------

def migrate():

    validate_configuration()

    workspaces, users, logs = read_sqlite_data()

    expected = {
        "workspaces": len(workspaces),
        "users": len(users),
        "audit_logs": len(logs)
    }

    print("\nConnecting to PostgreSQL...")

    # The connection context commits on success
    # and rolls back automatically if an error occurs.
    with psycopg.connect(DATABASE_URL) as pg:

        create_postgres_tables(pg)

        check_empty_tables(pg)

        insert_records(pg, workspaces, users, logs)

        reset_sequences(pg)

        verify_migration(pg, expected)

    print("\nMigration completed successfully!")


if __name__ == "__main__":

    try:
        migrate()

    except Exception as error:
        print("\nMigration failed:")
        print(type(error).__name__, "-", error)
        raise
