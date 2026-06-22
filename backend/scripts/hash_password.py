#!/usr/bin/env python3
"""
hash_password.py — Generate a bcrypt password hash for ADMIN_PASSWORD_HASH.

Usage:
    python backend/scripts/hash_password.py

Then copy the printed hash into your .env file, keeping the single-quotes:
    ADMIN_PASSWORD_HASH='<paste hash here>'

IMPORTANT: Always enclose the hash in SINGLE-QUOTES in the .env file.
Bcrypt hashes contain $ characters that shells expand to empty strings when
the value is unquoted, which causes an "Invalid salt" error at login.
"""

import getpass
import sys

try:
    import bcrypt
except ImportError:
    sys.exit(
        "bcrypt is not installed.  Run:  pip install bcrypt\n"
        "Or, from the backend directory:  pip install -r requirements.txt"
    )


def main() -> None:
    print("=== Admin Password Hash Generator ===\n")
    password = getpass.getpass("Enter new admin password: ")
    if len(password) < 8:
        sys.exit("Error: password must be at least 8 characters.")

    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        sys.exit("Error: passwords do not match.")

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
    hash_str = hashed.decode("utf-8")

    print("\n✅  Hash generated successfully.\n")
    print("Copy the line below (with single-quotes) into your .env file:\n")
    print(f"ADMIN_PASSWORD_HASH='{hash_str}'\n")
    print(
        "⚠️  The single-quotes are REQUIRED — bcrypt hashes start with $2b$ and\n"
        "    shell variable expansion will corrupt them if left unquoted.\n"
    )


if __name__ == "__main__":
    main()
