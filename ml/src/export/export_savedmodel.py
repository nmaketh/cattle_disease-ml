from pathlib import Path

from src.utils.io import read_yaml


def main() -> int:
    cfg = read_yaml("ml/configs/config.yaml")
    artifact_root = Path(cfg["paths"]["artifacts_dir"]).resolve()
    src_saved = artifact_root / "image_model"
    if not src_saved.exists():
        raise FileNotFoundError(f"Image model directory not found: {src_saved}")
    if not (src_saved / "saved_model.pb").exists():
        raise FileNotFoundError(f"saved_model.pb not found under {src_saved}. Train image model first.")

    print(f"[ok] SavedModel already available at {src_saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
