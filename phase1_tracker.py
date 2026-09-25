import cv2
import numpy as np 
import supervision as sv
from ultralytics import YOLO

PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32

def main(video_path: str, output_path: str):

    model = YOLO("yolov8x.pt")

    tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5, text_thickness=1, text_padding=5
    )

    video_info = sv.VideoInfo.from_video_path(video_path)
    frame_generator = sv.get_video_frames_generator(source_path=video_path)
    with sv.VideoSink(
        target_path=output_path, video_info=video_info
    ) as writer:
        for frame_idx, frame in enumerate(frame_generator):

            results = model(frame, verbose=False)[0]

            detections = sv.Detections.from_ultralytics(results)

            detections = detections[
                (detections.class_id == PERSON_CLASS_ID)
                |(detections.class_id == BALL_CLASS_ID)
            ]

            detections = tracker.update_with_detections(detections)

            labels = []
            for tracker_id, class_id in zip(
                detections.tracker_id, detections.class_id
            ):
                class_name = "Ball" if class_id == BALL_CLASS_ID else "Player"
                if tracker_id is not None:
                    labels.append(f"#{tracker_id} {class_name}")
                else:
                    labels.append(f"{class_name}")

            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            writer.write_frame(annotated_frame)


            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx}/{video_info.total_frames} frames...")

    print(f"Tracking complete. Video saved to: {output_path}")

if __name__ == "__main__":
    INPUT_VIDEO = "sample_video.mp4"
    OUTPUT_VIDEO = "output_tracked.mp4"

    main(INPUT_VIDEO, OUTPUT_VIDEO)