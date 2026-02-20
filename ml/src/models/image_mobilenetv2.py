from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


@dataclass
class ImageModelConfig:
    img_size: int = 224
    num_classes: int = 3
    dropout: float = 0.3


def build_image_model(cfg: ImageModelConfig) -> Tuple[Model, Model]:
    inputs = layers.Input(shape=(cfg.img_size, cfg.img_size, 3), name="image")
    x = preprocess_input(inputs)
    base = MobileNetV2(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.Dropout(cfg.dropout)(x)
    outputs = layers.Dense(cfg.num_classes, activation="softmax", name="probs")(x)

    model = Model(inputs=inputs, outputs=outputs, name="mobilenetv2_normal_lsd_fmd")
    return model, base


def compile_for_head(model: Model, lr: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def compile_for_finetune(model: Model, lr: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_top_layers(base_model: Model, fine_tune_at: int) -> None:
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
