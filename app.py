import streamlit as st
import PIL
import cv2
import numpy
import utils
import io

# 비디오 재생 함수
def play_video(video_source):
    camera = cv2.VideoCapture(video_source)

    st_frame = st.empty()
    while(camera.isOpened()):
        ret, frame = camera.read()

        if ret:
             visualized_image = utils.predict_image(frame, conf_threshold)
             st_frame.image(visualized_image, channels = "BGR")
        
        else:
            camera.release()
            break

# 기본 형태
st.set_page_config(
    page_title="Age/Gender/Emotion",
    page_icon=":sun_with_face:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Age/Gender/Emotion :sun_with_face:") # 제목

st.sidebar.header("Type") # 사이드바
source_radio = st.sidebar.radio("Select Source", ["IMAGE", "VIDEO", "WEBCAM"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the Cofidence Threshold", 10, 100, 20))/100

input = None

if source_radio == "IMAGE": # 이미지 업로드
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an image.", type = ("jpg", "png"))

    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv = cv2.cvtColor(numpy.array(uploaded_image), cv2.COLOR_RGB2BGR)
        visualized_image = utils.predict_image(uploaded_image_cv, conf_threshold)

        st.image(visualized_image, channels="BGR")

    else:
        st.image("data/sample_image.jpg")
        st.write("Click on 'Browse Files' in the sidebar to run inference on an image.")

temprary_location = None # 변수 초기화

if source_radio == "VIDEO": # 비디오 업로드
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an Video.", type = ("mp4"))

    if input is not None:
        g = io.BytesIO(input.read())
        temprary_location = "upload.mp4"

        with open(temprary_location, "wb") as out:
            out.write(g.read())

        out.close()

    if temprary_location is not None:
        play_video(temprary_location)
        if st.button("Replay", type="primary"):
            pass # 비디오 다시재생

    else:
        st.video("data/sample_video.mp4")
        st.write("Click on 'Browse Files' in the sidebar to run inference on an video.")

if source_radio == "WEBCAM":
    play_video(0)



