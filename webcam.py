import cv2
from collections import defaultdict
import json
from git import Repo
import numpy as np
import time
from datetime import datetime
import os
from ultralytics import YOLO
import math
import requests
from environ import weather_api_key, location


response = requests.get(
    f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
)
weather_data = response.json()
print(weather_data["weather"][0]["description"], weather_data["main"]["temp"])

cap = cv2.VideoCapture(0)
cap.set(3, 2560)
cap.set(4, 1920)

# Load YOLO model
model = YOLO("yolo11s.pt")  # load an official model

# Store the track history
track_history = defaultdict(lambda: [])
# Load class names
#with open(coco_names, "r") as f:
#    class_names = f.read().strip().split("\n")

#net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

# Function to detect objects
def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    output_layers = net.getUnconnectedOutLayersNames()
    detections = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_names[class_id] == "car":
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(boxes[i], confidences[i]) for i in indexes.flatten()]

# Function to determine traffic light color
red_light_roi = (50, 50, 100, 100)
green_light_roi = (50, 150, 100, 200)

def detect_light_color(frame):
    red_roi = frame[red_light_roi[1]:red_light_roi[3], red_light_roi[0]:red_light_roi[2]]
    green_roi = frame[green_light_roi[1]:green_light_roi[3], green_light_roi[0]:green_light_roi[2]]

    red_mean = np.mean(cv2.cvtColor(red_roi, cv2.COLOR_BGR2GRAY))
    green_mean = np.mean(cv2.cvtColor(green_roi, cv2.COLOR_BGR2GRAY))

    if red_mean > green_mean:
        return "red"
    elif green_mean > red_mean:
        return "green"
    else:
        return "unknown"


def commit_and_push_to_github(repo_path, filename):
    try:
        repo = Repo(repo_path)
        repo.git.add(filename)
        repo.index.commit(f"Update data {datetime.utcnow().isoformat()}")
        origin = repo.remote(name='origin')
        origin.push()
        print("Data pushed to GitHub.")
    except Exception as e:
        print("Error pushing to GitHub:", e)

def save_data_to_json(hour_data):
    current_time = datetime.utcnow().replace(
        minute=0, second=0, microsecond=0
    ).isoformat() + "Z"

    weather_data = get_weather_data_stub()  
    new_entry = {
        "timestamp": current_time,
        "aggregates": hour_data,
        "weather": weather_data
    }

    data_file = "traffic_data.json"
    if not os.path.exists(data_file):
        with open(data_file, "w") as f:
            json.dump([new_entry], f, indent=2)
    else:
        with open(data_file, "r") as f:
            existing_data = json.load(f)
        existing_data.append(new_entry)
        with open(data_file, "w") as f:
            json.dump(existing_data, f, indent=2)

    # Optional GitHub push
    commit_and_push_to_github("/path/to/your/local/repo", data_file)


# Function to check if cars are blocking the crosswalk or box
intersection_box = (200, 200, 400, 300)
crosswalk_box = (200, 300, 400, 350)

def is_blocking(frame, detections):
    violations = 0
    for (box, confidence) in detections:
        x, y, w, h = box
        car_box = (x, y, x + w, y + h)

        if (car_box[0] < intersection_box[2] and car_box[2] > intersection_box[0] and
            car_box[1] < intersection_box[3] and car_box[3] > intersection_box[1]):
            violations += 1

        if (car_box[0] < crosswalk_box[2] and car_box[2] > crosswalk_box[0] and
            car_box[1] < crosswalk_box[3] and car_box[3] > crosswalk_box[1]):
            violations += 1

    return violations


while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs from the results
        if (len(results) == 0 or results[0].boxes is None):
            continue
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            #vect = points[len(points)] - points[0]
            
            # determine start and end points of the line to determine which cardinal direction the line is pointing and where it started and ended      
            if len(points) > 1:
                start = points[0][0]
                end = points[-1][0]
                vect = end - start

                # determine the angle of the line
                angle = math.atan2(vect[1], vect[0]) * 180 / math.pi
                # determine the cardinal direction of the line
                if angle < 0:
                    angle += 360
                if angle < 45 or angle > 315:
                    direction = "E"
                elif angle < 135:
                    direction = "N"
                elif angle < 225:
                    direction = "W"
                else:
                    direction = "S"
                
                # draw the line
                cv2.line(annotated_frame, tuple(start), tuple(end), (0, 0, 255), 2)
                # put the direction text
                cv2.putText(annotated_frame, direction, tuple(end), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            #start = points[0]
            #end = points[len(points)]
            # determine the angle of the line
            #angle = math.atan2(vect[1], vect[0]) * 180 / math.pi
            # determine the cardinal direction of the line
            #if angle < 0:
            #    angle += 360
            #if angle < 45 or angle > 315:
            #    direction = "E"
            #elif angle < 135:
            #    direction = "N"
            #elif angle < 225:
            #    direction = "W"
            #else:
            #    direction = "S"
            # draw the line
            #cv2.line(annotated_frame, tuple(start), tuple(end), (0, 0, 255), 2)
        
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()