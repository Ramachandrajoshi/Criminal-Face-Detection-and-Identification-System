"""
Compare embedding versions (DeepFace ArcFace 1–3 vs InsightFace 4) on a
labelled face dataset — accuracy, detection rate, latency.

Dataset layout (LFW-style, one folder per person, ≥ 2 images for genuine pairs):

    DATASET_DIR/
        alice/  001.jpg  002.jpg ...
        bob/    001.jpg  ...

Usage (inside the backend container, or any env with requirements installed):

    python scripts/benchmark_engines.py /data/faces --versions 3 4
    python scripts/benchmark_engines.py /data/faces --probe-transform backlight
    python scripts/benchmark_engines.py /data/lfw --lfw-pairs /data/pairs.txt --probe-transform cctv

Reports, per version:
  * detection rate                — images where a face was found
  * genuine / impostor distances  — same-person vs different-person cosine distance
  * FAR / TAR at the configured threshold, and the threshold that gives
    FAR = 1% / 0.1% on this data (AGENTS.md §11 targets FAR ≤ 0.1%)
  * identification                — first image per person enrolled, the rest
    searched: correct accepts, false accepts, misses (what the app does)
  * latency p50 / p95 per image (detect → embed, warm)
  * with --lfw-pairs: the official LFW 10-fold verification accuracy (the
    protocol behind published "99.8%" figures); the second image of each
    pair goes through --probe-transform

Use the suggested thresholds as input to MATCH_THRESHOLD_INSIGHTFACE only
after the bias audit (AGENTS.md §8). Impostor FAR below 1 / impostor-pair
count is not measurable — use thousands of identities for a 0.1% estimate.

No images or embeddings are written anywhere; only aggregate numbers.
"""

import argparse
import base64
import itertools
import json
import os
import secrets
import sys
import time
from pathlib import Path

import numpy as np

# Settings validation needs these; the benchmark never touches the DB or API.
os.environ.setdefault("JWT_SECRET", secrets.token_urlsafe(16))
os.environ.setdefault("POSTGRES_PASSWORD", secrets.token_urlsafe(16))
os.environ.setdefault("DB_ENCRYPTION_KEY", base64.b64encode(secrets.token_bytes(32)).decode())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import (  # noqa: E402
    DEEPFACE_EMBEDDING_VERSIONS,
    INSIGHTFACE_EMBEDDING_VERSION,
    SUPPORTED_EMBEDDING_VERSIONS,
    match_threshold_for_version,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------- Dataset ----------
def load_dataset(root: Path, max_per_identity: int) -> dict[str, list[Path]]:
    people = {}
    for person in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(f for f in person.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
        if images:
            people[person.name] = images[:max_per_identity]
    return people


def transform_probe(image_bytes: bytes, kind: str) -> bytes:
    """Simulate poor capture conditions on probe images."""
    if kind == "none":
        return image_bytes
    import cv2

    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes

    def jpeg(out: np.ndarray, quality: int) -> bytes:
        return cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()

    # Sizes assume head-and-shoulders photos like LFW (250 px wide, ~45 px
    # between the eyes): "lowres" leaves ~14 px between the eyes, "cctv"
    # ~20 px, inside a 1080p frame.
    if kind in ("lowres", "cctv"):
        s = min(1.0, (80 if kind == "lowres" else 120) / max(img.shape[:2]))
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if kind == "lowres":
            return jpeg(small, 70)
        frame = np.full((1080, 1920, 3), 110, np.uint8)
        frame[200:200 + small.shape[0], 300:300 + small.shape[1]] = small
        return jpeg(frame, 80)
    if kind == "blur":
        return jpeg(cv2.GaussianBlur(img, (0, 0), 2.0), 40)

    x = img.astype(np.float32) / 255.0
    dark = 0.35 * x ** 2.6                         # heavily under-exposed subject
    if kind == "dark":
        out = dark
    else:                                          # "backlight": dark subject, blown-out surround
        h, w = x.shape[:2]
        yy, xx = np.mgrid[:h, :w]
        r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
        glow = np.clip(r - 0.5, 0, 1)[..., None]
        out = np.clip(dark + glow * 1.2, 0, 1)
    ok, buf = cv2.imencode(".jpg", (out * 255).astype(np.uint8))
    return buf.tobytes()


# ---------- Embedding ----------
def make_embedder(version: int, enrollment: bool = False):
    """
    Return bytes → embedding | None for one version, using the app's own
    pipeline code. ``enrollment`` picks faces the way registration does.
    """
    from app.core import pipeline

    if version == INSIGHTFACE_EMBEDDING_VERSION:
        return lambda image_bytes: pipeline.embed_with_insightface(image_bytes, enrollment)

    def deepface_embed(image_bytes: bytes):
        aligned = pipeline.detect_face(image_bytes)
        return None if aligned is None else pipeline.extract_embedding(aligned, version)

    return deepface_embed


def embed_all(version: int, items: list[tuple[str, bytes]], enrollment: bool = False) -> tuple[list, list[float]]:
    embed = make_embedder(version, enrollment)
    if items:
        embed(items[0][1])                         # warm-up: model load is not latency
    embeddings, latencies = [], []
    for _, data in items:
        start = time.perf_counter()
        emb = embed(data)
        latencies.append((time.perf_counter() - start) * 1000)
        embeddings.append(emb)
    return embeddings, latencies


# ---------- Metrics ----------
def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def verification_metrics(labels, embeddings, threshold, impostor_pairs, rng) -> dict:
    valid = [i for i, e in enumerate(embeddings) if e is not None]
    by_person: dict[str, list[int]] = {}
    for i in valid:
        by_person.setdefault(labels[i], []).append(i)

    genuine = [
        cosine_distance(embeddings[a], embeddings[b])
        for idx in by_person.values() for a, b in itertools.combinations(idx, 2)
    ]
    impostor = []
    people = list(by_person)
    if len(people) >= 2:
        for _ in range(impostor_pairs):
            p1, p2 = rng.choice(len(people), size=2, replace=False)
            a = rng.choice(by_person[people[p1]])
            b = rng.choice(by_person[people[p2]])
            impostor.append(cosine_distance(embeddings[a], embeddings[b]))

    genuine_arr, impostor_arr = np.array(genuine), np.array(impostor)
    out = {
        "genuine_pairs": len(genuine),
        "impostor_pairs": len(impostor),
        "threshold": threshold,
    }
    if genuine:
        out["genuine_distance_mean"] = float(genuine_arr.mean())
        out["genuine_distance_p95"] = float(np.percentile(genuine_arr, 95))
        out["tar_at_threshold"] = float((genuine_arr <= threshold).mean())
    if impostor:
        out["impostor_distance_mean"] = float(impostor_arr.mean())
        out["impostor_distance_p01"] = float(np.percentile(impostor_arr, 1))
        out["far_at_threshold"] = float((impostor_arr <= threshold).mean())
        for far in (0.01, 0.001):
            # Largest threshold whose impostor accept rate stays ≤ target.
            t = float(np.quantile(impostor_arr, far))
            out[f"threshold_at_far_{far}"] = t
            if genuine:
                out[f"tar_at_far_{far}"] = float((genuine_arr <= t).mean())
    return out


def identification_metrics(labels, gallery_idx, probe_idx, gallery_emb, probe_emb, threshold) -> dict:
    gallery = [(labels[i], e) for i, e in zip(gallery_idx, gallery_emb) if e is not None]
    correct = false_accept = miss = no_face = 0
    own_distances = []
    for i, emb in zip(probe_idx, probe_emb):
        if emb is None:
            no_face += 1
            continue
        own = [cosine_distance(emb, g) for lab, g in gallery if lab == labels[i]]
        if own:
            own_distances.append(own[0])
        best_label, best_dist = min(
            ((lab, cosine_distance(emb, g)) for lab, g in gallery), key=lambda p: p[1]
        )
        if best_dist > threshold:
            miss += 1
        elif best_label == labels[i]:
            correct += 1
        else:
            false_accept += 1
    total = max(1, len(probe_idx))
    return {
        "probes": len(probe_idx),
        "correct_accept_rate": correct / total,
        "false_accept_rate": false_accept / total,
        "miss_rate": miss / total,
        "no_face_rate": no_face / total,
        # Distance from each probe to its own enrolment photo — how far a
        # capture condition pushes a person away from their stored vector.
        "probe_to_own_enrolment_distance_mean":
            float(np.mean(own_distances)) if own_distances else None,
    }


def read_lfw_pairs(path: Path) -> list[tuple[str, str, int, int]]:
    """LFW pairs.txt → [(image_a, image_b, same_person, fold)], paths relative to the dataset."""
    lines = path.read_text().strip().splitlines()
    per_fold = int(lines[0].split()[1])
    pairs = []
    for i, line in enumerate(lines[1:]):
        p = line.split()
        fold = i // (2 * per_fold)
        if len(p) == 3:
            pairs.append((f"{p[0]}/{p[0]}_{int(p[1]):04d}.jpg", f"{p[0]}/{p[0]}_{int(p[2]):04d}.jpg", 1, fold))
        else:
            pairs.append((f"{p[0]}/{p[0]}_{int(p[1]):04d}.jpg", f"{p[2]}/{p[2]}_{int(p[3]):04d}.jpg", 0, fold))
    return pairs


def lfw_pair_metrics(version: int, root: Path, pairs, probe_transform: str) -> dict:
    """
    Official LFW protocol: per fold, the threshold that is best on the other
    nine folds is applied to this one. A pair with no face found counts as
    "different people".
    """
    enroll, search = make_embedder(version, enrollment=True), make_embedder(version)
    first = {a: enroll((root / a).read_bytes()) for a in {p[0] for p in pairs}}
    second = {b: search(transform_probe((root / b).read_bytes(), probe_transform)) for b in {p[1] for p in pairs}}
    sims = np.array([
        -1.0 if first[a] is None or second[b] is None else 1.0 - cosine_distance(first[a], second[b])
        for a, b, _, _ in pairs
    ])
    same = np.array([p[2] for p in pairs])
    folds = np.array([p[3] for p in pairs])
    accuracies = []
    for k in np.unique(folds):
        train, test = folds != k, folds == k
        order = np.argsort(-sims[train])
        y = same[train][order]
        # Accuracy when every pair at or above each candidate threshold is "same".
        acc = (np.cumsum(y) + ((1 - y).sum() - np.cumsum(1 - y))) / len(y)
        threshold = sims[train][order][np.argmax(acc)]
        accuracies.append(float(((sims[test] >= threshold) == same[test]).mean()))
    return {
        "pairs": len(pairs),
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "no_face": sum(e is None for e in first.values()) + sum(e is None for e in second.values()),
    }


def gpu_memory_used_mb():
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return [int(x) for x in out.stdout.split()] if out.returncode == 0 else None
    except (OSError, ValueError):
        return None


# ---------- Main ----------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--versions", type=int, nargs="+", default=[3, INSIGHTFACE_EMBEDDING_VERSION],
                        choices=SUPPORTED_EMBEDDING_VERSIONS)
    parser.add_argument("--max-per-identity", type=int, default=10)
    parser.add_argument("--impostor-pairs", type=int, default=20000)
    parser.add_argument("--probe-transform", choices=["none", "dark", "backlight", "lowres", "blur", "cctv"],
                        default="none", help="degrade probe (non-gallery) images to simulate poor capture")
    parser.add_argument("--lfw-pairs", type=Path,
                        help="LFW pairs.txt: also report the official 10-fold verification accuracy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="write the full report as JSON")
    args = parser.parse_args()

    people = load_dataset(args.dataset, args.max_per_identity)
    if not people:
        parser.error(f"no identity folders with images under {args.dataset}")

    labels, raw = [], []
    for person, files in people.items():
        for f in files:
            labels.append(person)
            raw.append(f.read_bytes())

    # First image per person is the enrolment photo; the rest are probes.
    seen, gallery_idx, probe_idx = set(), [], []
    for i, lab in enumerate(labels):
        (probe_idx if lab in seen else gallery_idx).append(i)
        seen.add(lab)
    probe_bytes = {i: transform_probe(raw[i], args.probe_transform) for i in probe_idx}

    print(f"{len(people)} identities, {len(labels)} images, probe transform: {args.probe_transform}")
    report = {"dataset": str(args.dataset), "identities": len(people), "images": len(labels),
              "probe_transform": args.probe_transform, "versions": {}}

    for version in args.versions:
        engine = "insightface" if version == INSIGHTFACE_EMBEDDING_VERSION else "deepface"
        print(f"\n── version {version} ({engine}) ──")
        rng = np.random.default_rng(args.seed)
        threshold = match_threshold_for_version(version)

        clean_emb, lat = embed_all(version, [(labels[i], raw[i]) for i in range(len(raw))], enrollment=True)
        probe_emb, _ = embed_all(version, [(labels[i], probe_bytes[i]) for i in probe_idx])

        result = {
            "engine": engine,
            "detection_rate": sum(e is not None for e in clean_emb) / len(clean_emb),
            "latency_ms_p50": float(np.percentile(lat, 50)),
            "latency_ms_p95": float(np.percentile(lat, 95)),
            "verification": verification_metrics(labels, clean_emb, threshold, args.impostor_pairs, rng),
            "identification": identification_metrics(
                labels, gallery_idx, probe_idx,
                [clean_emb[i] for i in gallery_idx], probe_emb, threshold,
            ),
            "gpu_memory_used_mb": gpu_memory_used_mb(),
        }
        if version == INSIGHTFACE_EMBEDDING_VERSION:
            from app.core.face_engine import get_engine
            result["providers"] = get_engine().providers
        if args.lfw_pairs:
            result["lfw_pairs"] = lfw_pair_metrics(
                version, args.dataset, read_lfw_pairs(args.lfw_pairs), args.probe_transform
            )
        report["versions"][version] = result

        v, ident = result["verification"], result["identification"]
        print(f"detection rate        {result['detection_rate']:.1%}")
        print(f"latency p50 / p95     {result['latency_ms_p50']:.0f} / {result['latency_ms_p95']:.0f} ms")
        if "genuine_distance_mean" in v:
            print(f"genuine dist mean/p95 {v['genuine_distance_mean']:.3f} / {v['genuine_distance_p95']:.3f}")
        if "impostor_distance_mean" in v:
            print(f"impostor dist mean/p1 {v['impostor_distance_mean']:.3f} / {v['impostor_distance_p01']:.3f}")
        if "far_at_threshold" in v:
            print(f"@ threshold {threshold:.2f}      TAR {v.get('tar_at_threshold', float('nan')):.1%}  "
                  f"FAR {v['far_at_threshold']:.3%}")
            for far in (0.01, 0.001):
                print(f"@ FAR {far:<6}         threshold {v[f'threshold_at_far_{far}']:.3f}  "
                      f"TAR {v.get(f'tar_at_far_{far}', float('nan')):.1%}")
        print(f"identification        correct {ident['correct_accept_rate']:.1%}  "
              f"false-accept {ident['false_accept_rate']:.1%}  miss {ident['miss_rate']:.1%}  "
              f"no-face {ident['no_face_rate']:.1%}")
        if ident["probe_to_own_enrolment_distance_mean"] is not None:
            print(f"probe→own enrolment   mean distance {ident['probe_to_own_enrolment_distance_mean']:.3f}")
        if result.get("providers"):
            print(f"providers             {result['providers']}")
        if "lfw_pairs" in result:
            lp = result["lfw_pairs"]
            print(f"LFW 10-fold accuracy  {lp['accuracy_mean']:.2%} ± {lp['accuracy_std']:.2%}  "
                  f"({lp['pairs']} pairs, {lp['no_face']} images with no face)")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {args.out}")


if __name__ == "__main__":
    main()
