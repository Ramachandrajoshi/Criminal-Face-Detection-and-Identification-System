"""
seed_synthetic.py — Generate 100,000 synthetic face-embedding profiles
for development and performance testing.

Usage:
    python seed_synthetic.py --count 100000

Requires:
    - PostgreSQL running locally on default port
    - psycopg2 or asyncpg installed
"""

import argparse
import hashlib
import os
import random
import sys
from datetime import datetime, timezone

import numpy as np

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)


# ---------- configuration ----------
DEFAULT_COUNT = 100_000
EMBEDDING_DIM = 512
FIRST_NAMES = [
    "James", "Maria", "Wei", "Aisha", "Carlos", "Fatima", "Olga", "Raj",
    "Sophie", "Hassan", "Mei", "Diego", "Yuki", "Ivan", "Priya", "Ahmed",
    "Elena", "Kenji", "Nadia", "Samuel", "Lina", "Omar", "Chloe", "Viktor",
    "Zara", "Marco", "Anya", "Ravi", "Clara", "Dmitri", "Amara", "Luis",
    "Hana", "Nikolai", "Isabella", "Kwame", "Mia", "Takeshi", "Sofia",
]
LAST_NAMES = [
    "Smith", "Garcia", "Wang", "Patel", "Mueller", "Tanaka", "Ivanov",
    "Kim", "Silva", "Ali", "Johnson", "Chen", "Singh", "Martinez", "Park",
    "Andersen", "Moreau", "Bose", "O'Brien", "Nakamura", "Dimitrov",
    "Santos", "Popov", "Al-Rashid", "Fischer", "Zhang", "Obi", "Larsen",
]
ALIAS_PREFIXES = ["Unknown", "Alias", "aka"]
ETHNICITIES = ["Caucasian", "East_Asian", "South_Asian", "African", "Hispanic", "Middle_Eastern"]
GENDERS = ["M", "F", "non-binary"]


def generate_embedding() -> np.ndarray:
    """Generate a unit-norm random 512-d vector."""
    vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec


def generate_demographics() -> dict:
    age_band = random.choice(["18-35", "36-60", "60+"])
    gender = random.choice(GENDERS)
    ethnicity = random.choice(ETHNICITIES)
    return {"age_band": age_band, "gender": gender, "ethnicity": ethnicity}


def seed(count: int = DEFAULT_COUNT):
    password = os.getenv("POSTGRES_PASSWORD")
    if not password:
        print("ERROR: POSTGRES_PASSWORD is required.")
        sys.exit(1)

    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "criminaldb"),
        user=os.getenv("POSTGRES_USER", "appuser"),
        password=password,
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
    )
    conn.autocommit = True
    cur = conn.cursor()

    batch_size = 1000
    total_batches = count // batch_size + 1

    print(f"Inserting {count} synthetic profiles in batches of {batch_size} …")

    for batch_idx in range(total_batches):
        rows = []
        for _ in range(batch_size):
            first = random.choice(FIRST_NAMES)
            last = random.choice(LAST_NAMES)
            name = f"{first} {last}"
            alias = random.choice(ALIAS_PREFIXES) if random.random() < 0.3 else None
            demographics = generate_demographics()
            embedding = generate_embedding()
            row = (name, alias, str(demographics).replace("'", '"'), embedding)
            rows.append(row)

        cur.executemany(
            """
            INSERT INTO suspect_profiles (suspect_name, alias, demographics, face_embedding)
            VALUES (%s, %s, %s::jsonb, %s::vector)
            """,
            rows,
        )
        conn.commit()
        print(f"  Batch {batch_idx + 1}/{total_batches} committed ({len(rows)} rows)")

    # Verify
    cur.execute("SELECT count(*) FROM suspect_profiles;")
    total = cur.fetchone()[0]
    print(f"\nDone. Total rows in suspect_profiles: {total}")

    # Rebuild HNSW index (it was created during init.sql; rebuild covers any gaps)
    print("Rebuilding HNSW index …")
    cur.execute("REINDEX INDEX suspect_embedding_hnsw_idx;")
    conn.commit()
    print("HNSW index rebuilt.")
    cur.close()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed synthetic suspect profiles")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of profiles")
    args = parser.parse_args()
    seed(args.count)
