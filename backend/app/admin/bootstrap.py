"""Create the first owner account.

    docker exec -it semantic_api python -m backend.app.admin.bootstrap \
        --email you@czargroup.net --name "Your Name"

Interactive on purpose. The password is read from a TTY with getpass and is
never a command-line argument, because argv is visible in `ps` to every process
on the box and lands in shell history on the way there. There is no --password
flag and adding one would undo that.

This exists because there is a chicken-and-egg: creating an admin_user requires
being an owner, and no owner exists. The alternatives were seeding an account in
a migration — a password hash committed to git, which is a credential in git —
or leaving the break-glass key as the permanent way in, which
ADMIN_CONSOLE_PLAN.md §13.14 is explicit about not doing.

Refuses to create a second owner unless --allow-additional is passed: run twice
by mistake and you get a clear error, not a surprise extra account with
owner rights and a password whoever ran it has forgotten.
"""
from __future__ import annotations

import argparse
import getpass
import sys

from sqlalchemy import text

from backend.app.admin import auth_service
from backend.app.services.database import SessionLocal


def create_first_owner(
    db,
    email: str,
    name: str,
    password: str,
    allow_additional: bool = False,
) -> str:
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        raise ValueError(f"{email!r} is not an email address.")
    if not (name or "").strip():
        raise ValueError("A name is required.")

    problem = auth_service.validate_password_strength(password)
    if problem:
        raise ValueError(problem)

    existing_owners = db.execute(
        text("SELECT COUNT(*) FROM admin_users WHERE role = 'owner' AND is_active = 1")
    ).scalar()
    if existing_owners and not allow_additional:
        raise RuntimeError(
            f"{existing_owners} active owner account(s) already exist. "
            f"Create further admins through the console, or pass "
            f"--allow-additional if you are certain."
        )

    if db.execute(
        text("SELECT COUNT(*) FROM admin_users WHERE email = :e"), {"e": email}
    ).scalar():
        raise RuntimeError(f"An account already exists for {email}.")

    admin_id = auth_service.new_admin_id()
    db.execute(
        text("""
            INSERT INTO admin_users (id, email, name, password_hash, role, is_active)
            VALUES (:id, :email, :name, :hash, 'owner', 1)
        """),
        {
            "id": admin_id,
            "email": email,
            "name": name.strip(),
            "hash": auth_service.hash_password(password),
        },
    )
    db.commit()
    return admin_id


def _read_password() -> str:
    """Prompt twice, confirm they match, check strength before asking again."""
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Not a TTY — run with `docker exec -it`, not `docker exec`. "
            "The password is read interactively and is never taken as an "
            "argument, so it cannot be piped in either."
        )
    for _ in range(3):
        first = getpass.getpass("Password: ")
        problem = auth_service.validate_password_strength(first)
        if problem:
            print(f"  {problem}")
            continue
        if first != getpass.getpass("Confirm password: "):
            print("  Passwords do not match.")
            continue
        return first
    raise RuntimeError("Giving up after three attempts.")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Create the first admin owner account.")
    parser.add_argument("--email", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--allow-additional",
        action="store_true",
        help="Create an owner even though one already exists.",
    )
    args = parser.parse_args(argv)

    db = SessionLocal()
    try:
        password = _read_password()
        admin_id = create_first_owner(
            db, args.email, args.name, password, allow_additional=args.allow_additional
        )
    except (RuntimeError, ValueError) as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    finally:
        db.close()

    print(f"\nCreated owner {args.email} ({admin_id}).")
    print("Sign in at /admin, then remove AICHATBOT_OPERATOR_KEY from the "
          "production .env — while it is set, RBAC and the audit trail are "
          "decorative (ADMIN_CONSOLE_PLAN.md §13.14).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
