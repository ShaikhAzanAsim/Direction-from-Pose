import cv2
import torch
import numpy as np
from ultralytics import YOLO
from core.container import setup_dependencies

from services.model_service import ModelService
from constants.models import HELMET_MODEL, YOLO_V8_N_POSE
import asyncio
from host.datasource.redis_client import redis_client
import os


class HelmetDetectorLive:
    def __init__(self, video_source):
        self.video_source = video_source
        self.pose_model = None
        self.helmet_model = None

    async def initialize_models(self):
        """Initialize both models asynchronously using ModelService"""
        await redis_init()
        model_service1 = ModelService(YOLO_V8_N_POSE)
        model_service2 = ModelService(HELMET_MODEL)

        await model_service1.initialize_model(YOLO_V8_N_POSE)
        await model_service2.initialize_model(HELMET_MODEL)

        # ✅ Now retrieve models from service properties
        self.pose_model = model_service1.model
        self.helmet_model = model_service2.model

    @staticmethod
    def point_in_box(box, point):
        x1, y1, x2, y2 = box
        px, py = point
        return x1 <= px <= x2 and y1 <= py <= y2

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print("❌ Error: Could not open video stream.")
            return

        print("✅ Running helmet detection... Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ End of video stream.")
                break

            # Run helmet detection
            helmet_results = self.helmet_model(frame, verbose=False)
            helmet_boxes = []
            for r in helmet_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    helmet_boxes.append([x1, y1, x2, y2])
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, "Helmet", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            helmet_found = len(helmet_boxes) > 0

            # Run pose detection
            pose_results = self.pose_model(frame, verbose=False)

            for r in pose_results:
                for kps in r.keypoints.xy:
                    head_points = kps[0:5]
                    head_x = float(torch.mean(head_points[:, 0]))
                    head_y = float(torch.mean(head_points[:, 1]))
                    head_point = (head_x, head_y)
                    cv2.circle(frame, (int(head_x), int(head_y)), 5, (255, 0, 0), -1)

                    helmet_on = any(self.point_in_box(hb, head_point) for hb in helmet_boxes)

                    if helmet_on:
                        label = "Person Wearing Helmet"
                        color = (0, 255, 0)
                    elif helmet_found:
                        label = "Person Not Wearing Helmet"
                        color = (0, 0, 255)
                    else:
                        label = "No Helmet Found"
                        color = (0, 255, 255)

                    cv2.putText(frame, label, (int(head_x) - 70, int(head_y) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Helmet Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


async def main():
    video_url = "https://ai-public-videos.s3.us-east-2.amazonaws.com/Inferenced+Videos/fall_and_helmet.mp4"
    detector = HelmetDetectorLive(video_url)
    await detector.initialize_models()
    detector.run()

async def redis_init():
    await redis_client.connect(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD", "Verseye!23")
    )
    await redis_client.set("test", "test")

if __name__ == "__main__":

    setup_dependencies()
    asyncio.run(main())





### ITH MULTI MODEL ###


import cv2
import torch
import asyncio
import numpy as np
from ultralytics import YOLO
from core.container import setup_dependencies, container
from host.datasource.redis_client import redis_client
from services.interfaces.imodel_service import IModelService
from constants.models import HELMET_MODEL, YOLO_V8_N_POSE
import os


class HelmetDetectorLive:
    def __init__(self, video_source: str):
        self.video_source = video_source
        self.pose_model: YOLO = None
        self.helmet_model: YOLO = None

    async def initialize_models(self):
        """Initialize both models concurrently using ModelService and DI"""
        await redis_init()

        # ✅ Resolve IModelService from DI container (interface, not concrete class)
        model_service_pose: IModelService = container.resolve(IModelService)
        model_service_helmet: IModelService = container.resolve(IModelService)

        # ✅ Initialize both models in parallel
        await asyncio.gather(
            model_service_pose.initialize_model(YOLO_V8_N_POSE),
            model_service_helmet.initialize_model(HELMET_MODEL)
        )

        # ✅ Get model instances
        self.pose_model = model_service_pose.model
        self.helmet_model = model_service_helmet.model

        print("✅ Both YOLO models initialized successfully!")

    @staticmethod
    def point_in_box(box, point):
        x1, y1, x2, y2 = box
        px, py = point
        return x1 <= px <= x2 and y1 <= py <= y2

    async def process_frame(self, frame):
        """Run both models in parallel for a single frame"""
        if self.pose_model is None or self.helmet_model is None:
            raise RuntimeError("Models are not initialized")

        # Run both models concurrently
        task_pose = asyncio.to_thread(self.pose_model, frame, verbose=False)
        task_helmet = asyncio.to_thread(self.helmet_model, frame, verbose=False)

        pose_results, helmet_results = await asyncio.gather(task_pose, task_helmet)

        helmet_boxes = []
        for r in helmet_results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                helmet_boxes.append([x1, y1, x2, y2])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, "Helmet", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        helmet_found = len(helmet_boxes) > 0

        for r in pose_results:
            for kps in r.keypoints.xy:
                head_points = kps[0:5]
                head_x = float(torch.mean(head_points[:, 0]))
                head_y = float(torch.mean(head_points[:, 1]))
                head_point = (head_x, head_y)
                cv2.circle(frame, (int(head_x), int(head_y)), 5, (255, 0, 0), -1)

                helmet_on = any(self.point_in_box(hb, head_point) for hb in helmet_boxes)

                if helmet_on:
                    label = "Person Wearing Helmet"
                    color = (0, 255, 0)
                elif helmet_found:
                    label = "Person Not Wearing Helmet"
                    color = (0, 0, 255)
                else:
                    label = "No Helmet Found"
                    color = (0, 255, 255)

                cv2.putText(frame, label, (int(head_x) - 70, int(head_y) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

    async def run(self):
        """Main loop: read video frames and process asynchronously"""
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print("❌ Error: Could not open video stream.")
            return

        print("✅ Running helmet detection... Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ End of video stream.")
                break

            processed_frame = await self.process_frame(frame)
            cv2.imshow("Helmet Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


async def redis_init():
    """Ensure Redis connection before model loading"""
    await redis_client.connect(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD", "Verseye!23")
    )
    await redis_client.set("test", "test")
    print("✅ Redis connected successfully!")


async def main():
    setup_dependencies()
    video_url = "https://ai-public-videos.s3.us-east-2.amazonaws.com/Inferenced+Videos/fall_and_helmet.mp4"
    detector = HelmetDetectorLive(video_url)
    await detector.initialize_models()
    await detector.run()


if __name__ == "__main__":
    asyncio.run(main())