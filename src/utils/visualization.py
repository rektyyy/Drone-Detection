"""Plotting helpers for qualitative analysis of detection results."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.utils import draw_bounding_boxes

# One fixed color per class so figures are consistent across models.
CLASS_COLORS = {
    "AirPlane": "red",
    "Drone": "lime",
    "Helicopter": "deepskyblue",
}


def draw_boxes(
    image: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    scores: torch.Tensor | None = None,
) -> Image.Image:
    """Draw labeled bounding boxes on an image and return a PIL image.

    Args:
        image:       (3, H, W) float tensor in [0, 1] or uint8 tensor in [0, 255].
        boxes_xyxy:  (N, 4) tensor of [x1, y1, x2, y2] in pixel coordinates.
        labels:      (N,) integer class indices.
        class_names: maps class index -> name (also used to pick a color).
        scores:      optional (N,) confidence scores, appended to the label text.
    """
    img = image.detach().cpu()
    if img.dtype != torch.uint8:
        img = (img.clamp(0, 1) * 255).to(torch.uint8)

    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return Image.fromarray(img.permute(1, 2, 0).numpy())

    boxes = boxes_xyxy.detach().cpu()
    labels = labels.detach().cpu()
    text_labels, colors = [], []
    for i, lbl in enumerate(labels.tolist()):
        name = class_names[lbl] if 0 <= lbl < len(class_names) else str(lbl)
        text = name if scores is None else f"{name} {scores[i].item():.2f}"
        text_labels.append(text)
        colors.append(CLASS_COLORS.get(name, "yellow"))

    drawn = draw_bounding_boxes(img, boxes, labels=text_labels, colors=colors, width=2)
    return Image.fromarray(drawn.permute(1, 2, 0).numpy())


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Confusion matrix",
    normalize: bool = False,
) -> plt.Figure:
    """Render a confusion matrix as an annotated heatmap.

    ``cm`` is expected to be a square matrix whose rows are ground-truth classes and
    columns are predicted classes. Extra trailing row/column (e.g. a "background"/
    missed-detection bucket) are supported by passing matching ``class_names``.
    """
    cm = np.asarray(cm, dtype=float)
    display = cm.copy()
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, out=np.zeros_like(display), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(display, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)

    thresh = display.max() / 2 if display.max() > 0 else 0.5
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            text = f"{display[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if display[i, j] > thresh else "black")

    fig.tight_layout()
    return fig
