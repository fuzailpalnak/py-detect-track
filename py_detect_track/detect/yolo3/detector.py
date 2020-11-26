import colorsys
import os
import random
from typing import List

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw

from py_detect_track.detect.yolo3.evaluator import get_evaluator, YOLOModelEvaluator


def letterbox_image(image, size):
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w * 1.0 / image_w, h * 1.0 / image_h))
    new_h = int(image_h * min(w * 1.0 / image_w, h * 1.0 / image_h))
    resized_image = image.resize((new_w, new_h), Image.BICUBIC)

    boxed_image = Image.new("RGB", size, (128, 128, 128))
    boxed_image.paste(resized_image, ((w - new_w) // 2, (h - new_h) // 2))
    return boxed_image


class YOLODetector(object):
    def __init__(
        self,
        model_path: str = None,
        anchors_path: str = None,
        classes_path: str = None,
        class_to_detect: str = "person",
        score_threshold: float = 0.60,
        iou_threshold: float = 0.60,
        model_image_size: tuple = (416, 416),
    ):

        self._class_to_detect = class_to_detect
        self._score = score_threshold
        self._iou = iou_threshold
        self._model_image_size = model_image_size

        self._class_names = self._get_class(classes_path)
        self._anchors = self._get_anchors(anchors_path)
        self._is_fixed_size = self._model_image_size != (None, None)

        self._model = load_model(os.path.expanduser(model_path), compile=False)

        self._sess = K.get_session()
        self._input_image_shape = K.placeholder(shape=(2,))

        self._evaluator = self._generate_evaluator()

    @staticmethod
    def _get_class(class_path: str) -> List:
        classes_path = os.path.expanduser(class_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    @staticmethod
    def _get_anchors(anchors_path: str) -> np.ndarray:
        anchors_path = os.path.expanduser(anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(",")]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def _generate_evaluator(self) -> YOLOModelEvaluator:
        colors = self._generate_colors()
        random.seed(10101)
        random.shuffle(colors)
        random.seed(None)

        evaluator = get_evaluator(
            self._model.output,
            self._anchors,
            len(self._class_names),
            self._input_image_shape,
            score_threshold=self._score,
            iou_threshold=self._iou,
        )
        return evaluator

    def _generate_colors(self) -> List:
        hsv_tuples = [
            (x / len(self._class_names), 1.0, 1.0)
            for x in range(len(self._class_names))
        ]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors)
        )
        return colors

    def detect_image(self, image):
        return_boxes = list()
        return_class_name = list()
        person_counter = 0

        if self._is_fixed_size:
            assert self._model_image_size[0] % 32 == 0, "Multiples of 32 required"
            assert self._model_image_size[1] % 32 == 0, "Multiples of 32 required"
            boxed_image = letterbox_image(
                image, tuple(reversed(self._model_image_size))
            )
        else:
            new_image_size = (
                image.width - (image.width % 32),
                image.height - (image.height % 32),
            )
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype="float32")
        image_data /= 255.0
        image_data = np.expand_dims(image_data, 0)

        out_boxes, out_scores, out_classes = self._sess.run(
            [self._evaluator.boxes, self._evaluator.scores, self._evaluator.classes],
            feed_dict={
                self._model.input: image_data,
                self._input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0,
            },
        )
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self._class_names[c]

            if predicted_class != self._class_to_detect:
                continue

            person_counter += 1
            box = out_boxes[i]
            x = int(box[1])
            y = int(box[0])
            w = int(box[3] - box[1])
            h = int(box[2] - box[0])
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0
            return_boxes.append([x, y, w, h])
            return_class_name.append([predicted_class])
        return return_boxes, return_class_name

    def close_session(self):
        self._sess.close()
