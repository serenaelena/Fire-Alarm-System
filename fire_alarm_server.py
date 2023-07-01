import os
import cv2
import geocoder
import datetime
import streamlit as st
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, messaging, db

SERVER_IP_ADDRESS = "10.10.20.116"  # change this to the server's IP address

def firebase_send_alert(message, image_url):
    g = geocoder.ip('me')
    # g = geocoder.ip('8.8.8.8')  # Google's DNS server, for testing
    message = messaging.Message(
        notification=messaging.Notification(
            title='Fire Alert',
            body=message,
            image=image_url,
        ),
        topic='fireAlarm',
        data={
            'lat': str(g.latlng[0]),
            'lng': str(g.latlng[1]),
        },
    )
    # Send a message to the devices subscribed to the provided topic.
    response = messaging.send(message)
    # Response is a message ID string.
    print('Successfully sent message:', response)


def write_to_firebase_db(fire_alert_json):
    ref = db.reference('/fire_alert')
    ref.push(fire_alert_json)


def get_firebase_db():
    ref = db.reference('/fire_alert')
    database = ref.get()
    if database is None:
        return []
    event_list = list(database.values())
    return event_list


def limit_firebase_db(limit=5000):
    if limit <= 0:
        raise Exception("Limit must be greater than 0.")
    ref = db.reference('/fire_alert')
    database = ref.get()
    if database is None or len(database) <= limit:
        return
    keys = list(database.keys())
    for key in keys[:len(keys) - limit]:
        ref.child(key).delete()


try:
    if not os.path.exists("static"):
        os.makedirs("static")
    cred = credentials.Certificate("fire-alarm-key.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://fire-alarm-65287-default-rtdb.firebaseio.com/'
        })
    st.set_page_config(page_title="Fire Alarm App", layout="wide", page_icon="./favicon.ico")
    st.title("Fire Alarm App")
    model = YOLO("best.pt")
    tab_live_stream, tab_history = st.tabs(["Live Stream Fire Monitoring", "Alert History"])
    with tab_live_stream:
        st.header("Live Stream Fire Monitoring")
        CAM_ID = st.text_input("Enter a live stream source (number for webcam, RTSP or HTTP(S) URL):", "0")
        if CAM_ID.isnumeric():
            CAM_ID = int(CAM_ID)
        col_run, col_stop = st.columns(2)
        run = col_run.button("Start Live Stream Processing")
        stop = col_stop.button("Stop Live Stream Processing")
        if stop:
            run = False
        is_fire = False
        FRAME_WINDOW = st.image([], width=1280)
        if run:
            cam = cv2.VideoCapture(CAM_ID)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            while True:
                ret, image = cam.read()
                if not ret:
                    st.error("Failed to capture stream from this camera stream. Please try again.")
                    break
                results = model.predict(image)
                image = results[0].plot()
                img_filename = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.jpg")
                img_url = f'http://{SERVER_IP_ADDRESS}:8501/app/static/{img_filename}'
                if results[0].boxes.shape[0] > 0:
                    if not is_fire:
                        is_fire = True
                        cv2.imwrite(f"./static/{img_filename}", image)
                        # get current date and time and format it with month in words
                        formatted_date = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")
                        firebase_send_alert(f'🔥A fire broke out in the area on {formatted_date}!', img_url)
                        print("Fire detected!")
                        fire_history_json = {
                            "event": "Fire detected",
                            "source": CAM_ID,
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "image_url": img_url,
                        }
                        write_to_firebase_db(fire_history_json)
                else:
                    if is_fire:
                        is_fire = False
                        cv2.imwrite(f"./static/{img_filename}", image)
                        # get current date and time and format it with month in words
                        formatted_date = datetime.datetime.now().strftime("%B %d, %Y at %H:%M")
                        firebase_send_alert(f'🚒The fire in the area was extinguished on {formatted_date}!', img_url)
                        print("Fire extinguished!")
                        fire_history_json = {
                            "event": "Fire extinguished",
                            "source": CAM_ID,
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "image_url": img_url,
                        }
                        write_to_firebase_db(fire_history_json)
                FRAME_WINDOW.image(image, channels="BGR", width=1280)
            cam.release()
    with tab_history:
        st.header("Alert History")
        limit_firebase_db()
        fire_history = get_firebase_db()
        if len(fire_history) == 0:
            st.write("No fire alert history yet.")
        else:
            fire_history = sorted(fire_history, key=lambda k: k['timestamp'], reverse=True)
            st.dataframe(fire_history)
except Exception as e:
    print(e)
    raise Exception(str(e))
