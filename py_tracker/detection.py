import numpy as np


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
