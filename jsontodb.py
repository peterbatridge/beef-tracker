import json
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
# Use the service account key file directly.
cred = credentials.Certificate("serviceAccountKey.json") 
firebase_admin.initialize_app(cred)

def load_json_to_firestore(json_file_path):
    """
    Reads traffic_data.json and inserts each entry into Firestore.
    Ensures no duplicates by checking for existing `from_timestamp` and `to_timestamp`.
    """

    # Initialize Firestore
    db = firestore.client()  # Remove project ID to use credentials from the certificate.
    collection_ref = db.collection("traffic-data")

    # Read the JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

    for entry in data:
        from_timestamp = entry["from_timestamp"]
        to_timestamp = entry["to_timestamp"]

        # Query Firestore for documents with matching from_timestamp and to_timestamp
        existing_docs = collection_ref.where("from_timestamp", "==", from_timestamp).where("to_timestamp", "==", to_timestamp).stream()

        # If no matching documents are found, insert the new entry
        if not any(existing_docs):
            collection_ref.add(entry)
            print(f"Inserted entry: {from_timestamp} - {to_timestamp}")
        else:
            print(f"Skipped duplicate entry: {from_timestamp} - {to_timestamp}")

    print("Finished processing traffic_data.json!")

if __name__ == "__main__":
    load_json_to_firestore("public/traffic_data.json")