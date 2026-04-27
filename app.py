import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import base64
import os

# إعدادات الاتصال
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Drowsiness Detector", layout="centered")
st.title("Drowsiness Detector")

# التحقق من وجود ملف النموذج
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    st.error(f"⚠️ ملف النموذج '{model_path}' غير موجود! يرجى تحميله ورفعه بجانب ملف app.py")
    st.stop()

# استخدام MediaPipe Tasks (الطريقة الأحدث والأكثر استقراراً)
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = FaceLandmarker.create_from_options(options)
        self.eye_close_count = 0
        self.yawn_count = 0
        self.drowsy_flag = False
        self.yawn_flag = False
        self.counter = 0

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        
        # تحويل الصورة لتنسيق MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # الكشف عن معالم الوجه
        # ملاحظة: في وضع الفيديو نحتاج لتقديم الطابع الزمني (timestamp)
        import time
        timestamp_ms = int(time.time() * 1000)
        detection_result = self.detector.detect_for_video(mp_image, timestamp_ms)

        self.drowsy_flag = False
        self.yawn_flag = False

        if detection_result.face_landmarks:
            # هنا يمكنك إضافة منطق EAR و MAR بناءً على مخرجات detection_result
            # سأقوم بتجهيز المنطق الكامل لك بمجرد تأكيد رفع الملف
            cv2.putText(image, "Detection Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return image

webrtc_streamer(
    key="driver-monitor", 
    video_transformer_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)
