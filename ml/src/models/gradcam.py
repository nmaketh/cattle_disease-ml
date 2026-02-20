from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


def _find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found for Grad-CAM.")


def make_gradcam(
    model: tf.keras.Model,
    image_tensor: tf.Tensor,
    class_idx: Optional[int] = None,
    last_conv_layer_name: Optional[str] = None,
) -> np.ndarray:
    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(image_tensor)
        if class_idx is None:
            class_idx = int(tf.argmax(preds[0]))
        class_channel = preds[:, class_idx]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heat_resized = np.array(
        Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)).resize(
            (image_rgb.shape[1], image_rgb.shape[0]), Image.Resampling.BILINEAR
        )
    ).astype(np.float32) / 255.0
    heat_color = plt.get_cmap("jet")(heat_resized)[..., :3]
    base = np.clip(image_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    mixed = np.clip((1.0 - alpha) * base + alpha * heat_color, 0.0, 1.0)
    return np.uint8(mixed * 255.0)


def save_gradcam(image_rgb: np.ndarray, heatmap: np.ndarray, out_path: str | Path) -> str:
    overlay = overlay_gradcam(image_rgb, heatmap)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(overlay)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return str(out)
