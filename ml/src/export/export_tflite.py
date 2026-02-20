from pathlib import Path

import tensorflow as tf

from src.utils.io import read_yaml


def main() -> int:
    cfg = read_yaml("ml/configs/config.yaml")
    artifact_root = Path(cfg["paths"]["artifacts_dir"]).resolve()
    saved_model_dir = artifact_root / "image_model"
    if not saved_model_dir.exists():
        raise FileNotFoundError(f"SavedModel directory not found: {saved_model_dir}")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    tflite = converter.convert()

    out = artifact_root / "image_model.tflite"
    out.write_bytes(tflite)
    print(f"[save] TFLite={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
