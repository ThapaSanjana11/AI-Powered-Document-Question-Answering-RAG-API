import requests
import time
import sys

BASE_URL = "http://127.0.0.1:8000"

def wait_for_server():
    print("Waiting for server to start and download the model...")
    for _ in range(60):
        try:
            # The report endpoint doesn't depend on the model being fully loaded if it's hit, 
            # but FastAPI won't accept requests until startup_event finishes.
            response = requests.get(f"{BASE_URL}/report", timeout=2)
            if response.status_code == 200:
                print("Server is up and ready!\n")
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(2)
    print("Server failed to start in time.")
    return False

def test_endpoints():
    print("--- TEST 1: GET /report ---")
    res = requests.get(f"{BASE_URL}/report")
    print(f"Status: {res.status_code}")
    print(res.json())

    print("\n--- TEST 2: POST /query (Empty DB) ---")
    res = requests.post(f"{BASE_URL}/query", json={"question": "test"})
    print(f"Status: {res.status_code}")
    print(res.json())

    print("\n--- TEST 3: POST /upload ---")
    with open("test_document.txt", "rb") as f:
        res = requests.post(f"{BASE_URL}/upload", files={"file": f})
    print(f"Status: {res.status_code}")
    print(res.json())

    print("\n--- TEST 4: POST /query (After upload) ---")
    res = requests.post(f"{BASE_URL}/query", json={"question": "What were the Q3 revenue results?"})
    print(f"Status: {res.status_code}")
    print(res.json())

if __name__ == "__main__":
    if wait_for_server():
        test_endpoints()
    else:
        sys.exit(1)
