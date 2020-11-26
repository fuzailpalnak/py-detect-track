import numpy as np

from dataclasses import dataclass
from functools import wraps, reduce
from typing import List, Tuple

import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from tensorflow import Tensor


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")


@dataclass
class YOLOModelEvaluator:
    boxes: list
    scores: list
    classes: list


def _head(
    feats: Tensor, anchors: np.ndarray, num_classes: int, input_shape: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """

    :param feats:
    :param anchors:
    :param num_classes:
    :param input_shape:
    :return:
    """
    num_anchors = len(anchors)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(
        K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1],
    )
    grid_x = K.tile(
        K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1],
    )
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5]
    )

    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probability = K.sigmoid(feats[..., 5:])

    box_xy = (box_xy + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = box_wh * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))

    return box_xy, box_wh, box_confidence, box_class_probability


def _correct_boxes(
    box_xy: Tensor, box_wh: Tensor, input_shape: Tensor, image_shape: Tensor
) -> Tensor:
    """

    :param box_xy:
    :param box_wh:
    :param input_shape:
    :param image_shape:
    :return:
    """
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2.0 / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_minimums = box_yx - (box_hw / 2.0)
    box_maximums = box_yx + (box_hw / 2.0)
    boxes = K.concatenate(
        [
            box_minimums[..., 0:1],  # y_min
            box_minimums[..., 1:2],  # x_min
            box_maximums[..., 0:1],  # y_max
            box_maximums[..., 1:2],  # x_max
        ]
    )

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def _boxes_and_scores(
    feats: Tensor,
    anchors: np.ndarray,
    num_classes: int,
    input_shape: Tensor,
    image_shape: Tensor,
) -> Tuple[Tensor, Tensor]:
    """

    :param feats:
    :param anchors:
    :param num_classes:
    :param input_shape:
    :param image_shape:
    :return:
    """
    box_xy, box_wh, box_confidence, box_class_probability = _head(
        feats, anchors, num_classes, input_shape
    )
    boxes = _correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probability
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def get_evaluator(
    outputs: List[Tensor],
    anchors: np.ndarray,
    num_classes: int,
    image_shape: Tensor,
    max_boxes: int = 20,
    score_threshold: float = 0.6,
    iou_threshold: float = 0.5,
) -> YOLOModelEvaluator:
    """

    :param outputs:
    :param anchors:
    :param num_classes:
    :param image_shape:
    :param max_boxes:
    :param score_threshold:
    :param iou_threshold:
    :return:
    """
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = K.shape(outputs[0])[1:3] * 32
    boxes = list()
    box_scores = list()

    boxes_collection = list()
    scores_collection = list()
    classes_collection = list()

    for l in range(3):
        _boxes, _box_scores = _boxes_and_scores(
            outputs[l],
            anchors[anchor_mask[l]],
            num_classes,
            input_shape,
            image_shape,
        )
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype="int32")

    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        nms_index = tf.image.non_max_suppression(
            class_boxes,
            class_box_scores,
            max_boxes_tensor,
            iou_threshold=iou_threshold,
        )

        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, "int32") * c

        boxes_collection.append(class_boxes)
        scores_collection.append(class_box_scores)
        classes_collection.append(classes)

    boxes_collection = K.concatenate(boxes_collection, axis=0)
    scores_collection = K.concatenate(scores_collection, axis=0)
    classes_collection = K.concatenate(classes_collection, axis=0)

    return YOLOModelEvaluator(boxes_collection, scores_collection, classes_collection)
