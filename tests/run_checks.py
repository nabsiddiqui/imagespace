#!/usr/bin/env python3
"""Repeatable ImageSpace acceptance contract.

Goal: this command exits 0 only after two seeded analysis runs are identical,
all provenance/label/noise invariants pass, and frontend/package builds succeed.
Artifacts remain under build/acceptance/ to make failures inspectable and loops fast.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "build" / "acceptance"
FIXTURE = WORK / "fixture"
CACHE = WORK / "cache"
OUT_A = WORK / "run-a"
OUT_B = WORK / "run-b"
OUT_VARIANT = WORK / "variant"


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"PASS  {message}")


def run(command, cwd=ROOT):
    printable = " ".join(map(str, command))
    print(f"\nRUN   {printable}")
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise AssertionError(f"command failed ({result.returncode}): {printable}")
    return result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_fixture(count):
    FIXTURE.mkdir(parents=True, exist_ok=True)
    for child in FIXTURE.iterdir():
        child.unlink()
    images = sorted(
        path for path in (ROOT / "test-data").iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )[:count]
    check(len(images) == count, f"fixture contains exactly {count} images")
    for image in images:
        shutil.copy2(image, FIXTURE / image.name)


def pipeline_command(output, seed, extra=()):
    return [
        sys.executable, str(ROOT / "scripts" / "imagespace.py"),
        "-i", str(FIXTURE), "-o", str(output),
        "--no-hd", "--thumb-size", "32", "--atlas-size", "512",
        "--quality", "35", "--seed", str(seed),
        "--pca-dims", "20", "--tsne-perplexity", "8",
        "--min-cluster-size", "3", "--hdbscan-min-samples", "2",
        "--hdbscan-selection-method", "leaf",
        "--cache-dir", str(CACHE), "--data-only", *extra,
    ]


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_output(
    output,
    expected_count,
    expected_pca=20,
    vocabulary="imagespace-art-v1",
    min_samples=2,
    selection_method="leaf",
):
    data = output / "data"
    required = [
        "analysis_config.json", "manifest.json", "data.bin",
        "metadata.csv", "neighbors.bin",
    ]
    for name in required:
        check((data / name).is_file(), f"{name} exists")

    config = load_json(data / "analysis_config.json")
    manifest = load_json(data / "manifest.json")
    check(config["schemaVersion"] == 1, "provenance schema version is 1")
    check(config["software"]["version"] == "1.3.0", "provenance software version is current")
    check(config["corpus"]["imageCount"] == expected_count, "provenance image count matches fixture")
    check(len(config["corpus"]["fingerprint"]) == 64, "corpus fingerprint is recorded")
    check(config["embedding"]["modelId"] == "openai/clip-vit-base-patch32", "CLIP model id is recorded")
    check(config["pca"]["dimensions"] == expected_pca, "effective PCA dimensions are recorded")
    check(0 < config["pca"]["explainedVariance"] <= 1, "PCA explained variance is valid")
    check(config["tsne"]["seed"] is not None, "t-SNE seed is recorded")
    check(config["hdbscan"]["minSamples"] == min_samples, "HDBSCAN min_samples flag is wired")
    check(config["hdbscan"]["selectionMethod"] == selection_method, "HDBSCAN selection method is wired")
    check(config["hdbscan"]["noisePolicy"] == "preserve-unassigned", "noise policy remains preserve-unassigned")
    check(config["labels"]["vocabularyId"] == vocabulary, "label vocabulary id is recorded")
    check(manifest["provenance"] == config, "manifest embeds the complete provenance record")

    with (data / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        headers = rows[0].keys() if rows else []
    check(len(rows) == expected_count, "metadata row count matches fixture")
    check("display_cluster" not in headers, "deprecated display_cluster is absent")
    csv_noise = sum(row.get("cluster") == "-1" for row in rows)
    stride = manifest["bytesPerImage"]
    raw = (data / "data.bin").read_bytes()
    check(len(raw) == expected_count * stride, "binary size matches manifest contract")
    binary_noise = sum(
        struct.unpack_from("<H", raw, index * stride + 22)[0] == 65535
        for index in range(expected_count)
    )
    check(binary_noise == csv_noise == config["hdbscan"]["noiseCount"], "CSV, binary, and provenance noise counts agree")
    check(b"display_cluster" not in raw, "binary contains no legacy display_cluster marker")

    labels_path = data / "cluster_labels.json"
    if labels_path.exists():
        labels = load_json(labels_path)
        for label in labels.values():
            check(len(label["top3"]) == 3, "each generated label stores top-3 evidence")
            check("uncertain" in label and "scoreMargin" in label, "each label stores uncertainty evidence")
    return config


def validate_forced_labels():
    spec = importlib.util.spec_from_file_location("imagespace_under_test", ROOT / "scripts" / "imagespace.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    embeddings = __import__("numpy").load(CACHE / "embeddings.npy")
    np = __import__("numpy")
    module._extract_clip_torch = lambda images, model_id: np.zeros((len(images), 7), dtype=np.float32)
    routed, backend = module.extract_embeddings(["fixture"], "example/custom-clip")
    check(routed.shape == (1, 7) and backend == "transformers", "custom CLIP model routes through its matching transformers backend")

    forced_clusters = np.arange(len(embeddings), dtype=np.int32) % 2
    vocab = module.load_label_vocabulary(ROOT / "tests" / "fixtures" / "test-labels.json")
    labels = module.generate_cluster_labels(
        embeddings, forced_clusters, vocabulary=vocab, uncertainty_threshold=0.02
    )
    allowed = {entry["text"] for entry in vocab["candidates"]}
    check(len(labels) == 2, "forced label test emits two cluster labels")
    for label in labels.values():
        check(len(label["top3"]) == 3, "custom vocabulary emits top-3 candidates")
        check(all(item["text"] in allowed for item in label["top3"]), "label candidates come only from override vocabulary")
        check(label["uncertaintyThreshold"] == 0.02, "label uncertainty threshold is applied")
        check(label["vocabularyId"] == "imagespace-test-labels-v1", "custom vocabulary id reaches label output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixture-size", type=int, default=48)
    args = parser.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    make_fixture(args.fixture_size)
    for output in (OUT_A, OUT_B, OUT_VARIANT):
        shutil.rmtree(output, ignore_errors=True)

    run(pipeline_command(OUT_A, args.seed))
    config_a = validate_output(OUT_A, args.fixture_size)

    shutil.copytree(OUT_A, OUT_B)
    run(pipeline_command(OUT_B, args.seed, ("--relayout",)))
    config_b = validate_output(OUT_B, args.fixture_size)
    check(sha256(OUT_A / "data" / "data.bin") == sha256(OUT_B / "data" / "data.bin"), "two seeded layouts are byte-identical")
    check(config_a == config_b, "two seeded provenance records are identical")

    shutil.copytree(OUT_A, OUT_VARIANT)
    run(pipeline_command(OUT_VARIANT, args.seed, (
        "--relayout", "--pca-dims", "10",
        "--hdbscan-min-samples", "3", "--hdbscan-selection-method", "eom",
        "--label-candidates", str(ROOT / "tests" / "fixtures" / "test-labels.json"),
        "--label-uncertainty-threshold", "0.02",
    )))
    validate_output(
        OUT_VARIANT,
        args.fixture_size,
        expected_pca=10,
        vocabulary="imagespace-test-labels-v1",
        min_samples=3,
        selection_method="eom",
    )
    check(sha256(OUT_A / "data" / "data.bin") != sha256(OUT_VARIANT / "data" / "data.bin"), "PCA flag changes downstream layout bytes")
    validate_forced_labels()

    run([sys.executable, "-m", "py_compile", "scripts/imagespace.py"])
    run(["npm", "run", "build"], cwd=ROOT / "image_space")
    bundle_text = "".join(path.read_text(errors="ignore") for path in (ROOT / "image_space" / "output" / "assets").glob("*.js"))
    check("Methods & Provenance" in bundle_text, "frontend bundle contains the Methods panel")
    check("analysis_config.json" in bundle_text, "frontend bundle fetches analysis_config.json")

    wheel_dir = WORK / "wheel"
    shutil.rmtree(wheel_dir, ignore_errors=True)
    wheel_dir.mkdir()
    run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(wheel_dir)])
    wheel = next(wheel_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        members = archive.namelist()
    check(any(name.endswith("imagespace_resources/label_presets/art-v1.json") for name in members), "wheel contains bundled label vocabulary")

    print("\nSUCCESS: acceptance contract is green and seeded output is stable.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        raise
