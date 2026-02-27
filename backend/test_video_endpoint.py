import requests
import json
import time

def test_analyze_video():
    url = "http://127.0.0.1:8000/analyze-video"
    print(f"Testing {url}...")
    
    # We will just upload test_ai.jpg as a mock video file to see if the endpoint accepts it
    # We expect it to fail gracefully during video processing, or if we have a real short mp4 we'd use that.
    
    # Let's create a minimal dummy mp4 or just use an existing short file if we had one.
    # Since we don't have a guaranteed mp4, we will try to curl with the existing test_ai.jpg
    # to ensure the router at least parses the multiform data.
    files = {'file': ('test.mp4', open('test_ai.jpg', 'rb'), 'video/mp4')}
    
    start = time.time()
    try:
        response = requests.post(url, files=files)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
        
    print(f"Elapsed time: {time.time() - start:.2f}s")
    
if __name__ == "__main__":
    test_analyze_video()
