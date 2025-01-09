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
lost_tracks = {}
LOST_TRACK_THRESHOLD = 10
DOORWAY_REGION = (1100, 175, 1250, 325)  
DOORWAY_FRAMES_UNTIL_FINALIZED = 18000 # ~5mins
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

def is_in_doorway(x, y):
    x1, y1, x2, y2 = DOORWAY_REGION
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def get_current_weather():
    try:
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
            'description': 'failed to get weather data',
                'humidity': 0,
            }
    except Exception as e:
        print(e)
        return {
            'temp': 0.0,
            'feels_like': 0.0,
            'main': '',
            'description': 'failed to get weather data',
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

    data_file = "public/traffic_data.json"
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
        avg_conf = track_data['sum_conf'] / frames_seen
        if frames_seen >= LINGER_THRESHOLD_FRAMES and avg_conf >= 0.2:
            counters['people'] += 1
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

model = YOLO("yolo11x.pt")
current_weather = get_current_weather()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    now_local = datetime.now(chicago_tz)
    if now_local >= next_save_time:
        save_data_to_json()
        for k in counters:
            counters[k] = 0
        from_time = now_local.replace(microsecond=0)
        from_time_str = from_time.isoformat()
        current_weather = get_current_weather()
        next_save_time = from_time + timedelta(minutes=15)

    results = model.track(frame, persist=True, verbose=False, device='cuda', tracker="bytetrack.yaml")
    if (
        not results or 
        results[0].boxes is None or 
        results[0].boxes.id is None or 
        results[0].boxes.id.shape[0] == 0
    ):
        #cv2.imshow("YOLO Tracking", frame)
        if keyboard.is_pressed('i'):
            print(counters)
        elif keyboard.is_pressed('q'):
            break
        continue

    xywh = results[0].boxes.xywh.cpu().numpy()
    track_ids = results[0].boxes.id.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy()
    conf_ids = results[0].boxes.conf.cpu().numpy()

    annotated_frame = results[0].plot()
    current_ids = set()

    for i, box in enumerate(xywh):
        t_id = int(track_ids[i])
        c_id = int(class_ids[i])
        conf = conf_ids[i]
        label = model.names[c_id]

        x_center, y_center, w, h = box
        current_ids.add(t_id)

        if t_id not in tracked_objects:
            tracked_objects[t_id] = {
                'label': label,
                'positions': [(x_center, y_center)],
                'frames_seen': 1,
                'sum_conf': conf
            }
        else:
            tracked_objects[t_id]['positions'].append((x_center, y_center))
            tracked_objects[t_id]['frames_seen'] += 1
            tracked_objects[t_id]['sum_conf'] += conf

        if t_id in lost_tracks:
            del lost_tracks[t_id]

    new_lost_tracks = {}
    for tid, info in lost_tracks.items():
        if tid in current_ids:
            continue
        
        info['lost_count'] += 1

        last_positions = info['last_data']['positions']
        if last_positions:
            lx, ly = last_positions[-1]
            in_door = is_in_doorway(lx, ly)
        else:
            in_door = False

        if in_door:
            # Wait long if last known in the doorway
            if info['lost_count'] > (DOORWAY_FRAMES_UNTIL_FINALIZED):
                finalize_track(info['last_data'])
        else:
            if info['lost_count'] > LOST_TRACK_THRESHOLD:
                finalize_track(info['last_data'])
            else:
                # keep it in lost tracks
                new_lost_tracks[tid] = info

    lost_tracks = new_lost_tracks

    ended_ids = [tid for tid in tracked_objects if tid not in current_ids]

    for tid in ended_ids:
        last_data = tracked_objects[tid]
        del tracked_objects[tid]

        # Move them to lost_tracks or increment lost_count if already there
        if tid in lost_tracks:
            lost_tracks[tid]['lost_count'] += 1
        else:
            lost_tracks[tid] = {
                'last_data': last_data,
                'lost_count': 1
            }
    # Use if viewing image
    #cv2.imshow("Tracking", annotated_frame)
    #if cv2.waitKey(1) & 0xFF == ord('i'):
    #    print(counters)
    #elif cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    if keyboard.is_pressed('i'):
        print(counters)
    elif keyboard.is_pressed('q'):
        break

cap.release()
cv2.destroyAllWindows()

def view_rect_for_testing(region):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        x1, y1, x2, y2 = region
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.imshow("Test Rectangle", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()