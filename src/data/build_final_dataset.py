"""
Build the final dataset from selected-70-harmful and local_variation outputs.

For each model:
  - Reads ../PatchBench/data_processing/selected-70-harmful/{stem}.json (70 samples)
  - Reads ../PatchBench/data_processing/local_variation/{stem}.json (local prompts)
  - Splits 70 samples into 50 train / 20 validation
  - Writes to ../PatchBench/final_dataset/{stem}/:
      row_data.json       — all 70 samples (id, prompt, answer, label)
      test.json           — local variations for the 50 train samples
      validation.json     — local variations for the 20 validation samples
"""

import json
from pathlib import Path

SELECTED_DIR   = Path("../PatchBench/data_processing/selected-70-harmful")
VARIATION_DIR  = Path("../PatchBench/data_processing/local_variation")
WILDGUARD_DIR  = Path("../PatchBench/data_processing/wildguard_inference")
OUTPUT_BASE    = Path("../PatchBench/final_dataset")

TRAIN_SIZE = 50
VAL_SIZE   = 20


def load_json(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_model(stem: str) -> None:
    selected_path  = SELECTED_DIR  / f"{stem}.json"
    variation_path = VARIATION_DIR / f"{stem}.json"

    if not selected_path.exists():
        print(f"[SKIP] No selected file for {stem}")
        return

    samples = load_json(selected_path)

    train_samples = samples[:TRAIN_SIZE]
    val_samples   = samples[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]


    val_ids = {str(s["id"]) for s in val_samples}
    wildguard_path = WILDGUARD_DIR / f"{stem}.json"
    if wildguard_path.exists():
        wildguard_data = load_json(wildguard_path)
        row_data = [
            {"id": s["id"], "prompt": s["prompt"], "answer": s["answer"], "label": s["label"]}
            for s in wildguard_data
            if str(s["id"]) not in val_ids
        ]
    else:
        raise FileNotFoundError(f"No wildguard file for {stem}: {wildguard_path}")

    # test.json / validation.json — local variations if available, raw split otherwise
    if variation_path.exists():
        variations = load_json(variation_path)
        variation_by_id = {str(v["id"]): v for v in variations}

        def get_variations(split: list) -> list:
            result = []
            for s in split:
                sid = str(s["id"])
                if sid in variation_by_id:
                    result.append(variation_by_id[sid])
                else:
                    print(f"  [WARN] No local variation found for id {sid}")
            return result

        test_data = get_variations(train_samples)
        val_data  = get_variations(val_samples)
        print(f"[DONE] {stem}: {len(row_data)} samples → {len(test_data)} test, {len(val_data)} validation (with local variations)")
    else:
        raise FileNotFoundError(f"No local variation file for {stem}")

    out_dir = OUTPUT_BASE / stem
    save_json(row_data,  out_dir / "row_data.json")
    save_json(test_data, out_dir / "test.json")
    save_json(val_data,  out_dir / "validation.json")


if __name__ == "__main__":
    stems = [p.stem for p in sorted(SELECTED_DIR.glob("*.json"))]

    if not stems:
        print(f"No files found in {SELECTED_DIR}")
    else:
        print(f"Found {len(stems)} models: {stems}")
        for stem in stems:
            build_model(stem)
        print("\nAll models processed.")