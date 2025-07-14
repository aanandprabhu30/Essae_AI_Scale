#!/usr/bin/env python3
"""
AIScale Produce Recognition - RK3568 Inference Script
"""

import cv2
import numpy as np
from rknnlite.api import RKNNLite
import time

class ProduceRecognizer:
    def __init__(self, model_path, class_names):
        self.class_names = class_names
        self.rknn_lite = RKNNLite()

        # Load RKNN model
        print("📦 Loading RKNN model...")
        if self.rknn_lite.load_rknn(model_path) != 0:
            raise RuntimeError('❌ Failed to load RKNN model.')

        # Init runtime (no core_mask for RK3568)
        print("⚙️ Initializing runtime...")
        if self.rknn_lite.init_runtime() != 0:
            raise RuntimeError('❌ Failed to initialize runtime.')

        print("✅ RKNNLite ready.\n")

    def preprocess(self, image):
        """Preprocess image for RKNN inference"""
        img = cv2.resize(image, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('uint8')  # Important: must be uint8
        return img

    def predict(self, image):
        """Run inference and return top prediction"""
        input_data = self.preprocess(image)
        input_data = np.expand_dims(input_data, axis=0)  # Make it [1, 224, 224, 3]

        start_time = time.time()
        outputs = self.rknn_lite.inference(inputs=[input_data])
        elapsed = time.time() - start_time

        if outputs is None or len(outputs) == 0:
            raise RuntimeError('❌ Inference returned no outputs!')

        predictions = outputs[0].squeeze()
        class_idx = np.argmax(predictions)
        confidence = predictions[class_idx]

        return {
            'class': self.class_names[class_idx],
            'confidence': float(confidence),
            'inference_time': elapsed,
            'all_scores': predictions.tolist()
        }

    def __del__(self):
        self.rknn_lite.release()


if __name__ == "__main__":
    class_names = ['apple', 'banana', 'black_grapes', 'mango', 'orange', 'pomegranate']
    recognizer = ProduceRecognizer(
        '/home/linux/Documents/Essae_AI_Scale/aiscale_deployment/models/mobilenetv3_produce.rknn',
        class_names
    )

    # Open camera (adjust index if needed)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open camera.")
        exit(1)

    print("🎥 Camera opened. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame. Exiting...")
            break

        try:
            result = recognizer.predict(frame)
            label = f"{result['class']}: {result['confidence']:.2f}"
            time_info = f"Inference: {result['inference_time']*1000:.1f} ms"

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, time_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            print(f"❌ Error during inference: {e}")
            cv2.putText(frame, "Inference failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("AIScale Produce Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
