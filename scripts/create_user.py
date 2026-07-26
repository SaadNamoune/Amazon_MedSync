"""
Creates a MedSync user account. Matches how real clinical software is
provisioned -- a hospital's IT admin creates accounts for staff, there's
no public self-registration on a system that can query patient diagnostics.

Usage:
    python scripts/create_user.py --username dr.smith --password "..." --role clinician
"""
import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medsync.auth import hash_password  # noqa: E402
from medsync.db import User, SessionLocal, init_db  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default=None, help="Omit to be prompted (recommended)")
    parser.add_argument("--role", default="clinician", choices=["clinician", "admin"])
    args = parser.parse_args()

    password = args.password or getpass.getpass("Password: ")
    if not password:
        print("Password cannot be empty.")
        return

    init_db()
    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == args.username).first():
            print(f"User '{args.username}' already exists.")
            return
        user = User(username=args.username, hashed_password=hash_password(password), role=args.role)
        db.add(user)
        db.commit()
        print(f"Created user '{args.username}' (role={args.role})")
    finally:
        db.close()


if __name__ == "__main__":
    main()
