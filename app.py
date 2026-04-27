import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import base64

# إعدادات الاتصال
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Drowsiness Detector", layout="centered")
st.title("Drowsiness Detector")

# استدعاء الحلول بالطريقة الرسمية
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.eye_close_count = 0
        self.yawn_count = 0
        self.drowsy_flag = False
        self.yawn_flag = False
        self.counter = 0

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        self.drowsy_flag = False
        self.yawn_flag = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # منطق الكشف (EAR/MAR)
                # ... (نفس المنطق السابق)
                pass

        return image

# تشغيل الكاميرا
webrtc_streamer(
    key="driver-monitor", 
    video_transformer_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)
