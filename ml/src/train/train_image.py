import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score

from src.models.gradcam import make_gradcam, save_gradcam
from src.models.image_mobilenetv2 import (
    ImageModelConfig,
    build_image_model,
    compile_for_finetune,
    compile_for_head,
    unfreeze_top_layers,
)
from src.utils.io import ensure_dir, read_yaml, write_json
from src.utils.seed import set_seed
from src.utils.viz import save_confusion_matrix

LABEL_TO_ID = {"Normal": 0, "LSD": 1, "FMD": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def make_ds(df: pd.DataFrame, img_size: int, batch_size: int, training: bool, seed: int) -> tf.data.Dataset:
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

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.ignore_errors()
    if training:
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=max(1024, len(paths)), seed=seed, reshuffle_each_iteration=True)
        aug = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.06),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomContrast(0.1),
            ]
        )

        def _aug(img, label):
            return aug(img, training=True), label

        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.cache()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def class_weight_from_df(df: pd.DataFrame) -> Dict[int, float]:
    counts = df["label"].value_counts().to_dict()
    n = len(df)
    n_cls = len(LABEL_TO_ID)
    out = {}
    for lbl, idx in LABEL_TO_ID.items():
        c = max(1, int(counts.get(lbl, 0)))
        out[idx] = float(n / (n_cls * c))
    return out


def evaluate_text(y_true_ids: np.ndarray, y_pred_ids: np.ndarray) -> str:
    y_true = [ID_TO_LABEL[int(x)] for x in y_true_ids]
    y_pred = [ID_TO_LABEL[int(x)] for x in y_pred_ids]
    rep = classification_report(y_true, y_pred, labels=["Normal", "LSD", "FMD"], zero_division=0)
    return rep


def tune_class_bias(
    y_prob_val: np.ndarray,
    y_true_val_ids: np.ndarray,
    labels: Tuple[str, str, str] = ("Normal", "LSD", "FMD"),
) -> np.ndarray:
    # Learn multiplicative class bias on validation only to improve macro F1.
    best_f1 = -1.0
    best_w = np.ones(len(labels), dtype=float)
    y_true_lbl = np.array([labels[int(i)] for i in y_true_val_ids])

    grid = np.arange(0.4, 2.61, 0.1)
    for wn in grid:
        for wl in grid:
            for wf in grid:
                w = np.array([wn, wl, wf], dtype=float)
                pred_ids = np.argmax(y_prob_val * w[None, :], axis=1)
                pred_lbl = np.array([labels[int(i)] for i in pred_ids])
                m = float(f1_score(y_true_lbl, pred_lbl, labels=list(labels), average="macro", zero_division=0))
                if m > best_f1:
                    best_f1 = m
                    best_w = w
    return best_w


def main() -> int:
    cfg = read_yaml("ml/configs/config.yaml")
    set_seed(int(cfg["seed"]))

    manifest_path = Path(cfg["paths"]["splits_manifest_csv"]).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing splits manifest: {manifest_path}. Run merge_images.py first.")
    df = pd.read_csv(manifest_path)

    print(f"[paths] splits_manifest={manifest_path}")
    print(f"[count] total_rows={len(df)}")
    print(f"[count] per_class={df['label'].value_counts().to_dict()}")

    for split in ["train", "val", "test"]:
        if split not in set(df["split"].unique()):
            raise ValueError(f"Missing split '{split}' in manifest.")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    img_cfg = cfg["image"]
    ds_train = make_ds(train_df, img_cfg["img_size"], img_cfg["batch_size"], True, int(cfg["seed"]))
    ds_val = make_ds(val_df, img_cfg["img_size"], img_cfg["batch_size"], False, int(cfg["seed"]))
    ds_test = make_ds(test_df, img_cfg["img_size"], img_cfg["batch_size"], False, int(cfg["seed"]))

    model_cfg = ImageModelConfig(img_size=img_cfg["img_size"], num_classes=3)
    model, base = build_image_model(model_cfg)

    artifact_root = Path(cfg["paths"]["artifacts_dir"]).resolve()
    image_dir = ensure_dir(artifact_root / "image_model")
    reports_dir = ensure_dir(artifact_root / "reports")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=img_cfg["early_stop_patience"], restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=img_cfg["reduce_lr_patience"], min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(image_dir / "best.keras"), monitor="val_accuracy", mode="max", save_best_only=True),
    ]

    compile_for_head(model, img_cfg["lr_head"])
    hist_head = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=img_cfg["head_epochs"],
        callbacks=callbacks,
        class_weight=class_weight_from_df(train_df),
        verbose=2,
    )

    unfreeze_top_layers(base, img_cfg["fine_tune_at"])
    compile_for_finetune(model, img_cfg["lr_finetune"])
    hist_ft = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=img_cfg["finetune_epochs"],
        callbacks=callbacks,
        class_weight=class_weight_from_df(train_df),
        verbose=2,
    )

    # Persist SavedModel + test predictions
    model.export(str(image_dir))

    y_true = np.array(test_df["label"].map(LABEL_TO_ID).tolist(), dtype=int)
    y_prob = model.predict(ds_test, verbose=0)
    y_true_val = np.array(val_df["label"].map(LABEL_TO_ID).tolist(), dtype=int)
    y_prob_val = model.predict(ds_val, verbose=0)

    class_bias = tune_class_bias(y_prob_val, y_true_val)
    y_pred = np.argmax(y_prob * class_bias[None, :], axis=1)

    np.save(image_dir / "test_probs.npy", y_prob)
    np.save(image_dir / "test_true.npy", y_true)
    test_df.assign(pred=[ID_TO_LABEL[int(x)] for x in y_pred]).to_csv(image_dir / "test_predictions.csv", index=False)
    write_json(
        image_dir / "decision_calibration.json",
        {
            "labels": ["Normal", "LSD", "FMD"],
            "class_bias": [float(x) for x in class_bias.tolist()],
            "note": "Bias learned on validation split only; applied at decision stage to maximize macro F1.",
        },
    )

    report_text = evaluate_text(y_true, y_pred)
    (reports_dir / "image_report.txt").write_text(report_text, encoding="utf-8")

    y_true_lbl = [ID_TO_LABEL[int(x)] for x in y_true]
    y_pred_lbl = [ID_TO_LABEL[int(x)] for x in y_pred]
    save_confusion_matrix(y_true_lbl, y_pred_lbl, ["Normal", "LSD", "FMD"], reports_dir / "confusion_matrix_image.png", "Image Model CM")

    # Export 3 Grad-CAM examples per class from test split.
    grad_dir = ensure_dir(reports_dir / "gradcam_examples")
    sampled = (
        test_df.groupby("label", group_keys=False)
        .apply(lambda x: x.head(3))
        .reset_index(drop=True)
    )
    for i, row in sampled.iterrows():
        p = Path(row["filepath"])
        raw = tf.io.read_file(str(p))
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_cfg["img_size"], img_cfg["img_size"]])
        img = tf.cast(img, tf.float32)
        x = img[None, ...]
        class_idx = LABEL_TO_ID[row["label"]]
        heat = make_gradcam(model, x, class_idx=class_idx)
        save_gradcam(
            image_rgb=tf.cast(img, tf.uint8).numpy(),
            heatmap=heat,
            out_path=grad_dir / f"{row['label']}_{i}.png",
        )

    write_json(artifact_root / "label_map.json", {"image_labels": ["Normal", "LSD", "FMD"], "final_labels": cfg["labels"]["final_labels"]})

    summary = {
        "history_head": {k: [float(x) for x in v] for k, v in hist_head.history.items()},
        "history_finetune": {k: [float(x) for x in v] for k, v in hist_ft.history.items()},
        "test_report": report_text,
        "decision_class_bias": [float(x) for x in class_bias.tolist()],
    }
    (image_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[save] SavedModel={image_dir}")
    print(f"[save] report={reports_dir / 'image_report.txt'}")
    print(f"[save] cm={reports_dir / 'confusion_matrix_image.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
