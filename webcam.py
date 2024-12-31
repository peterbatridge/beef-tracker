import cv2
import json
import subprocess
import numpy as np
from datetime import datetime, timedelta, timezone
import os
from ultralytics import YOLO
import requests
import keyboard
from zoneinfo import ZoneInfo
from environ import weather_api_key, location

tracked_objects = {}

current_weather = {
    'temp': 0.0,
    'feels_like': 0.0,
    'main': '',
    'description': '',
    'humidity': 0,
}

counters = {
    'people': 0,
    'cyclists': 0,
    'northbound_traffic': 0,
    'southbound_traffic': 0,
    'people_lingering': 0
}

LINGER_THRESHOLD_FRAMES = 30
LINGER_DISTANCE_THRESHOLD = 40  
VEHICLE_MOVE_THRESHOLD = 40
chicago_tz = ZoneInfo("America/Chicago")

from_time = datetime.now(chicago_tz).replace(microsecond=0)
next_save_time = from_time + timedelta(minutes=15)
from_time_str = from_time.isoformat()

def get_current_weather():
    response = requests.get(
        f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={weather_api_key}&units=imperial"
    )
    weather_data = response.json()
    if weather_data.get('cod') == 200:
        return {
            'temp': weather_data["main"]["temp"],
            'feels_like': weather_data["main"]["feels_like"],
            'main': weather_data["weather"][0]["main"],
            'description': weather_data["weather"][0]["description"],
            'humidity': weather_data["main"]["humidity"],
        }
    else:
        print("FAILED:", weather_data)
        return {
            'temp': 0.0,
            'feels_like': 0.0,
            'main': '',
            'description': '',
            'humidity': 0,
        }

def commit_and_push_to_github(filename):
    try:
        subprocess.run(["git", "pull"])
        subprocess.run(["git", "add", filename])
        subprocess.run(["git", "commit", "-m", "Update data"])
        subprocess.run(["git", "push"])
        print("Data pushed to GitHub.")
    except Exception as e:
        print("Error pushing to GitHub:", e)

def save_data_to_json():
    global from_time_str, counters

    current_time = datetime.now(chicago_tz).replace(microsecond=0)
    to_time_str = current_time.isoformat()

    new_entry = {
        "from_timestamp": from_time_str,
        "to_timestamp": to_time_str,
        "counts": dict(counters),
        "weather": current_weather
    }

    data_file = "traffic_data.json"
    if not os.path.exists(data_file):
        with open(data_file, "w") as f:
            json.dump([new_entry], f, indent=2)
    else:
        if os.path.getsize(data_file) == 0:
            existing_data = []
        else:
            with open(data_file, "r") as f:
                existing_data = json.load(f)
        existing_data.append(new_entry)
        with open(data_file, "w") as f:
            json.dump(existing_data, f, indent=2)

    commit_and_push_to_github(data_file)

def finalize_track(track_data):
    label = track_data['label']
    positions = track_data['positions']
    frames_seen = track_data['frames_seen']

    if label == "person":
        counters['people'] += 1
        if frames_seen >= LINGER_THRESHOLD_FRAMES:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            if ((max(xs) - min(xs) < LINGER_DISTANCE_THRESHOLD) and
                (max(ys) - min(ys) < LINGER_DISTANCE_THRESHOLD)):
                counters['people_lingering'] += 1
    elif label in ["bicycle", "cyclist"]:
        counters['cyclists'] += 1
    elif label in ["car", "truck", "bus", "motorcycle"]:
        first_y = positions[0][1]
        last_y = positions[-1][1]
        delta_y = last_y - first_y
        if abs(delta_y) < VEHICLE_MOVE_THRESHOLD:
            # Do nothing, consider it parked
            pass
        elif last_y < first_y:
            counters['northbound_traffic'] += 1
        else:
            counters['southbound_traffic'] += 1

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

model = YOLO("yolo11m.pt")
current_weather = get_current_weather()
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    now_utc = datetime.now(chicago_tz)
    if now_utc >= next_save_time:
        save_data_to_json()
        for k in counters:
            counters[k] = 0
        from_time = now_utc.replace(microsecond=0)
        from_time_str = from_time.isoformat()
        current_weather = get_current_weather()
        next_save_time = from_time + timedelta(minutes=15)

    results = model.track(frame, iou=0.3, persist=True, verbose=False)
    if (
        not results or 
        results[0].boxes is None or 
        results[0].boxes.id is None or 
        results[0].boxes.id.shape[0] == 0
    ):
        #cv2.imshow("YOLO Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    xywh = results[0].boxes.xywh.cpu().numpy()
    track_ids = results[0].boxes.id.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()

    #annotated_frame = results[0].plot()
    current_ids = set()

    for i, box in enumerate(xywh):
        t_id = int(track_ids[i])
        c_id = int(class_ids[i])
        label = model.names[c_id]

        x_center, y_center, w, h = box
        current_ids.add(t_id)

        if t_id not in tracked_objects:
            tracked_objects[t_id] = {
                'label': label,
                'positions': [(x_center, y_center)],
                'frames_seen': 1
            }
        else:
            tracked_objects[t_id]['positions'].append((x_center, y_center))
            tracked_objects[t_id]['frames_seen'] += 1

    ended_ids = [tid for tid in tracked_objects if tid not in current_ids]
    for tid in ended_ids:
        finalize_track(tracked_objects[tid])
        del tracked_objects[tid]

    #cv2.imshow("YOLO Tracking", annotated_frame)
    if keyboard.is_pressed('i'):
        print(counters)
    elif keyboard.is_pressed('q'):
        break

cap.release()
cv2.destroyAllWindows()
