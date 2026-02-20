from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

from src.utils.io import read_yaml
from src.utils.viz import save_confusion_matrix

LABEL_TO_ID = {"Normal": 0, "LSD": 1, "FMD": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def make_ds(df: pd.DataFrame, img_size: int, batch_size: int) -> tf.data.Dataset:
    paths = df["filepath"].astype(str).tolist()
    labels = df["label"].map(LABEL_TO_ID).astype("int32").tolist()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)
        return img, label

    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main() -> int:
    cfg = read_yaml("ml/configs/config.yaml")
    manifest = Path(cfg["paths"]["splits_manifest_csv"]).resolve()
    image_model_dir = Path(cfg["paths"]["artifacts_dir"]).resolve() / "image_model"

    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest {manifest}")
    if not image_model_dir.exists():
        raise FileNotFoundError(f"Missing image SavedModel dir {image_model_dir}")

    df = pd.read_csv(manifest)
    test_df = df[df["split"] == "test"].copy()
    ds_test = make_ds(test_df, cfg["image"]["img_size"], cfg["image"]["batch_size"])

    model = tf.keras.layers.TFSMLayer(str(image_model_dir), call_endpoint="serve")

    probs = []
    y_true = []
    for x_batch, y_batch in ds_test:
        out = model(x_batch)
        if isinstance(out, dict):
            out = list(out.values())[0]
        probs.append(np.array(out))
        y_true.extend(y_batch.numpy().tolist())
    y_prob = np.concatenate(probs, axis=0)
    y_pred = np.argmax(y_prob, axis=1)

    true_lbl = [ID_TO_LABEL[int(x)] for x in y_true]
    pred_lbl = [ID_TO_LABEL[int(x)] for x in y_pred]

    rep = classification_report(true_lbl, pred_lbl, labels=["Normal", "LSD", "FMD"], zero_division=0)
    reports_dir = Path(cfg["paths"]["artifacts_dir"]).resolve() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    (reports_dir / "image_report.txt").write_text(rep, encoding="utf-8")
    save_confusion_matrix(true_lbl, pred_lbl, ["Normal", "LSD", "FMD"], reports_dir / "confusion_matrix_image.png", "Image Model CM")

    print(rep)
    print(f"[save] {reports_dir / 'image_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
