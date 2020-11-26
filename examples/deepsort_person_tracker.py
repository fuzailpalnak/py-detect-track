import os
import sys
import cv2
import numpy as np
from PIL import Image
from collections import deque


module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

from py_detect_track.track import DeepSort
from py_detect_track.detect.yolo3 import YOLODetector
from py_detect_track.detect.appearance.deepsort_identification import create_box_encoder
from py_detect_track.track.deepsort.deepsort import DeepSortDetection
from py_detect_track.detect.utils import (
    non_max_suppression,
    from_x_y_width_height_to_x_min_y_min_x_max_y_max,
    from_x_y_aspect_height_to_x_y_width_height,
)

IDENTIFICATION_SAVED_MODEL_WEIGHT = (
    r"D:\Cypherics\saved_weights\deep_sort\mars-small128.pb"
)

DETECTOR_MODEL_WEIGHT_PATH = r"D:\Cypherics\saved_weights\deep_sort\yolo.h5"
DETECTOR_ANCHOR_PATH = r"D:\Cypherics\saved_weights\deep_sort\yolo_anchors.txt"
DETECTOR_CLASS_PATH = r"D:\Cypherics\saved_weights\deep_sort\coco_classes.txt"

INPUT_VIDEO_FILE = r"D:\Cypherics\open_back\data\10\
vlc-record-2020-11-15-18h32m45s-2018-03-09.10-10-00.10-15-00.school.G299.mp4-.mp4"
OUTPUT_VIDEO_FILE = r"D:\Cypherics\open_back\out\test_4.avi"

NMS_OVER_LAP = 0.3

np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

PTS = [deque(maxlen=30) for _ in range(9999)]


def input_video_object():
    return cv2.VideoCapture(INPUT_VIDEO_FILE)


def output_video_object(width, height):
    return cv2.VideoWriter(
        OUTPUT_VIDEO_FILE, cv2.VideoWriter_fourcc(*"MJPG"), 15, (width, height)
    )


def run():
    detector = YOLODetector(
        model_path=DETECTOR_MODEL_WEIGHT_PATH,
        anchors_path=DETECTOR_ANCHOR_PATH,
        classes_path=DETECTOR_CLASS_PATH,
    )
    encoder = create_box_encoder(IDENTIFICATION_SAVED_MODEL_WEIGHT, batch_size=1)
    deep_sort = DeepSort(
        cascade_matching_threshold=0.5, iou_matching_threshold=0.50, max_age=30
    )

    input_video = input_video_object()
    output_video = output_video_object(int(input_video.get(3)), int(input_video.get(4)))

    fps = 0.0

    try:
        while cv2.waitKey(1):
            identified_object_count = int(0)
            tracker_id_collection = []

            ret, frame = input_video.read()
            if not ret:
                raise Exception
            image = Image.fromarray(frame[..., ::-1])

            detected_boxes, class_names = detector.detect_image(image)
            appearance_features = encoder(frame, detected_boxes)

            detections = [
                DeepSortDetection(
                    np.asarray(bbox, dtype=np.float),
                    np.asarray(feature, dtype=np.float32),
                )
                for bbox, feature in zip(detected_boxes, appearance_features)
            ]

            detected_boxes = np.array([d.detection for d in detections])
            scores = np.array([1 for d in detections])
            indices = non_max_suppression(detected_boxes, NMS_OVER_LAP, scores)
            detections = [detections[i] for i in indices]

            trackers = deep_sort.track_with_detection_object(detections)

            for det in detections:
                bbox = from_x_y_width_height_to_x_min_y_min_x_max_y_max(
                    det.detection
                ).reshape(4, 1)
                cv2.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (255, 255, 255),
                    2,
                )

            for track in trackers:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                tracker_id_collection.append(int(track.track_id))

                bbox = from_x_y_width_height_to_x_min_y_min_x_max_y_max(
                    from_x_y_aspect_height_to_x_y_width_height(
                        track.extract_position_from_state()
                    )
                ).reshape(4, 1)
                color = [
                    int(c)
                    for c in COLORS[
                        tracker_id_collection[identified_object_count] % len(COLORS)
                    ]
                ]

                cv2.rectangle(
                    frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    color,
                    3,
                )
                cv2.putText(
                    frame,
                    str(track.track_id),
                    (int(bbox[0]), int(bbox[1] - 50)),
                    0,
                    5e-3 * 150,
                    color,
                    2,
                )

                if len(class_names) > 0:
                    cv2.putText(
                        frame,
                        str(class_names[0]),
                        (int(bbox[0]), int(bbox[1] - 20)),
                        0,
                        5e-3 * 150,
                        color,
                        2,
                    )

                identified_object_count += 1
                center = (
                    int(((bbox[0]) + (bbox[2])) / 2),
                    int(((bbox[1]) + (bbox[3])) / 2),
                )
                PTS[track.track_id].append(center)
                cv2.circle(frame, center, 1, color, 5)

                for j in range(1, len(PTS[track.track_id])):
                    if (
                        PTS[track.track_id][j - 1] is None
                        or PTS[track.track_id][j] is None
                    ):
                        continue

                    thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                    cv2.line(
                        frame,
                        (PTS[track.track_id][j - 1]),
                        (PTS[track.track_id][j]),
                        color,
                        thickness,
                    )

            cv2.putText(
                frame,
                "Total Object Counter: " + str(len(set(tracker_id_collection))),
                (int(20), int(120)),
                0,
                5e-3 * 200,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "Current Object Counter: " + str(identified_object_count),
                (int(20), int(80)),
                0,
                5e-3 * 200,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "FPS: %f" % fps,
                (int(20), int(40)),
                0,
                5e-3 * 200,
                (0, 255, 0),
                3,
            )
            cv2.namedWindow("Person-Tracking", 0)
            cv2.resizeWindow("Person-Tracking", 2048, 2048)
            cv2.imshow("Person-Tracking", frame)

            output_video.write(frame)

        input_video.release()
        output_video.release()
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        input_video.release()
        output_video.release()
        cv2.destroyAllWindows()
        raise KeyboardInterrupt

    except Exception as ex:
        input_video.release()
        output_video.release()
        cv2.destroyAllWindows()
        raise ex


if __name__ == "__main__":
    run()
