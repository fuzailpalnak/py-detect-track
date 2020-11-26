import numpy as np
import cv2


def from_x_y_aspect_height_to_x_y_width_height(bbox: np.ndarray) -> np.ndarray:
    """

    :param bbox:
    :return:
    """
    assert bbox.ndim == 1 or bbox.ndim == 2, (
        "Expected bbox to be either '1' dim or '2' dim" "got %s",
        (bbox.ndim,),
    )
    assert bbox.shape == (4,) or bbox.shape == (4, 1), (
        "Expected bbox to have shape '[4, ]' or '[4, 1]'" "got %s",
        (bbox.shape,),
    )

    copy_bbox = bbox.copy()
    copy_bbox[2] *= copy_bbox[3]
    copy_bbox[:2] -= copy_bbox[2:] / 2
    return copy_bbox


def from_x_y_width_height_to_x_y_aspect_height(bbox: np.ndarray) -> np.ndarray:
    """

    :param bbox:
    :return:
    """
    assert bbox.ndim == 1 or bbox.ndim == 2, (
        "Expected bbox to be either '1' dim or '2' dim" "got %s",
        (bbox.ndim,),
    )
    assert bbox.shape == (4,) or bbox.shape == (4, 1), (
        "Expected bbox to have shape '[4, ]' or '[4, 1]'" "got %s",
        (bbox.shape,),
    )
    copy_bbox = bbox.copy()

    copy_bbox[:2] += copy_bbox[2:] / 2
    copy_bbox[2] /= copy_bbox[3]
    return copy_bbox


def from_x_y_width_height_to_x_min_y_min_x_max_y_max(bbox: np.ndarray) -> np.ndarray:
    """

    :return: (min x, min y, max x, max y)
    """
    assert bbox.ndim == 1 or bbox.ndim == 2, (
        "Expected bbox to be either '1' dim or '2' dim" "got %s",
        (bbox.ndim,),
    )
    assert bbox.shape == (4,) or bbox.shape == (4, 1), (
        "Expected bbox to have shape '[4, ]' or '[4, 1]'" "got %s",
        (bbox.shape,),
    )
    copy_bbox = bbox.copy()

    copy_bbox[2:] += copy_bbox[:2]
    return copy_bbox


def from_x_min_y_min_x_max_y_max_to_scale_aspect_ratio(bbox: np.ndarray) -> np.ndarray:
    """

    :param bbox:
    :return:
    """

    assert bbox.ndim == 1 or bbox.ndim == 2, (
        "Expected bbox to be either '1' dim or '2' dim" "got %s",
        (bbox.ndim,),
    )
    assert bbox.shape == (4,) or bbox.shape == (4, 1), (
        "Expected bbox to have shape '[4, ]' or '[4, 1]'" "got %s",
        (bbox.shape,),
    )

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r])


def from_scale_aspect_ratio_to_x_min_y_min_x_max_y_max(bbox: np.ndarray) -> np.ndarray:
    """

    :param bbox:
    :return:
    """
    assert bbox.ndim == 1 or bbox.ndim == 2, (
        "Expected bbox to be either '1' dim or '2' dim" "got %s",
        (bbox.ndim,),
    )
    assert bbox.shape == (4,) or bbox.shape == (4, 1), (
        "Expected bbox to have shape '[4, ]' or '[4, 1]'" "got %s",
        (bbox.shape,),
    )

    w = np.sqrt(bbox[2] * bbox[3])
    h = bbox[2] / w
    return np.array(
        [
            bbox[0] - w / 2.0,
            bbox[1] - h / 2.0,
            bbox[0] + w / 2.0,
            bbox[1] + h / 2.0,
        ]
    )


def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    """
    Suppress overlapping detections.

    Original code from [1]_ has been adapted to include confidence score.

    .. [1] http://www.pyimagesearch.com/2015/02/16/
           faster-non-maximum-suppression-python/

    :param boxes:
    :param max_bbox_overlap:
    :param scores:
    :return:
    """

    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        indexes = np.argsort(scores)
    else:
        indexes = np.argsort(y2)

    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[indexes[:last]])
        yy1 = np.maximum(y1[i], y1[indexes[:last]])
        xx2 = np.minimum(x2[i], x2[indexes[:last]])
        yy2 = np.minimum(y2[i], y2[indexes[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[indexes[:last]]

        indexes = np.delete(
            indexes, np.concatenate(([last], np.where(overlap > max_bbox_overlap)[0]))
        )

    return pick
