#!/usr/bin/env python3
"""NEW helper: select one controlled Figure 5(a) ablation configuration.

Copy into the repository's scripts/ directory. Run variants sequentially:
do not change shared JSON files while any training workers are running.
This helper does not launch training or modify downstream trainer settings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    Path("configs/model/shallow.json"),
    Path("configs/embed_model/clip_ViTB32.json"),
    Path("configs/trainer/cifar10/flexmatch/clip_ViTB32_shallow.json"),
]
VARIANTS = {
    "baseline": {},
    "linear": {0: {"model_name": "linear"}},
    "h128": {0: {"hidden_dim": 128}},
    "h1024": {0: {"hidden_dim": 1024}},
    "k1": {1: {"num_transform_seeds": 1}},
    "k3": {1: {"num_transform_seeds": 3}},
    "u1": {2: {"uratio": 1}},
    "u3": {2: {"uratio": 3}},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=[*VARIANTS, "restore"])
    parser.add_argument("--output-dir", type=Path,
                        help="save immutable configuration/source manifest here")
    args = parser.parse_args()
    backup = ROOT / "work/ablation_original_configs"
    paths = [backup / p for p in FILES]
    if any(p.exists() for p in paths) and not all(p.exists() for p in paths):
        parser.error(f"incomplete original-config backup: {backup}")
    if not all(p.exists() for p in paths):
        if args.variant == "restore":
            parser.error("no original configuration backup exists")
        original = [(ROOT / p).read_bytes() for p in FILES]
        model, embed, trainer = [json.loads(b) for b in original]
        expected = [(model.get("model_name"), "shallow"),
                    (model.get("num_hidden"), 2),
                    (model.get("hidden_dim", 512), 512),
                    (embed.get("num_transform_seeds"), 5),
                    (trainer.get("uratio"), 5), (trainer.get("lr"), .001),
                    (trainer.get("wd"), .001), (trainer.get("train_batch_size"), 128),
                    (trainer.get("max_epoch"), 20), (trainer.get("early_stop"), True)]
        if any(actual != wanted for actual, wanted in expected):
            parser.error("start from the audited baseline JSON files before first use")
        for path, content in zip(paths, original):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    original = [p.read_bytes() for p in paths]
    if args.variant == "restore":
        for relative, content in zip(FILES, original):
            (ROOT / relative).write_bytes(content)
        print("Restored the three original JSON files byte-for-byte.")
        return
    contents = dict(zip(FILES, original))
    for index, changes in VARIANTS[args.variant].items():
        value = json.loads(contents[FILES[index]])
        value.update(changes)
        contents[FILES[index]] = (json.dumps(value, indent=2) + "\n").encode()
    # Store enough evidence to detect altered source code, which the existing
    # training database's configuration fingerprint does not cover.
    tracked = sorted({*ROOT.glob("configs/**/*.json"), *ROOT.glob("LabelBench/**/*.py"),
                      *ROOT.glob("scripts/*.py"), ROOT / "requirements.txt"})
    hashes = {}
    for path in tracked:
        relative = path.relative_to(ROOT)
        data = contents.get(relative, path.read_bytes())
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {"variant": args.variant, "commit": commit, "sha256": hashes}
    output = (args.output_dir or ROOT / "results/ablation" / args.variant).resolve()
    manifest_path = output / "ablation_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            parser.error("output already has different code/configuration; choose a new output directory")
    elif (output / "figure5a.sqlite").exists():
        parser.error("existing database has no ablation manifest; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    for relative, content in contents.items():
        (ROOT / relative).write_bytes(content)
    # Include exact JSON bytes, not just hashes, for later resumption/review.
    for source in ROOT.glob("configs/**/*.json"):
        target = output / "config_snapshot" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Configured {args.variant}; run selection AND evaluate before changing variants.")
    print(f"Output directory: {output}")
    print("Downstream fine-tuning configuration unchanged.")


if __name__ == "__main__":
    main()
