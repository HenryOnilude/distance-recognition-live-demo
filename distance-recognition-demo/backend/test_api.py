"""API endpoint tests using FastAPI TestClient"""
from fastapi.testclient import TestClient
from main import app
import io
import numpy as np
import cv2
from PIL import Image

client = TestClient(app)

def create_test_image_bytes():
    """Create a JPEG image in bytes"""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.ellipse(image, (320, 240), (100, 130), 0, 0, 360, (200, 180, 150), -1)
    cv2.circle(image, (290, 220), 10, (50, 50, 50), -1)
    cv2.circle(image, (350, 220), 10, (50, 50, 50), -1)
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes

def test_root_endpoint():
    print("Testing GET / ...")
    response = client.get("/")
    assert response.status_code == 200
    print("✅ Root endpoint OK")

def test_analyze_frame_success():
    print("Testing POST /analyze-frame...")
    img_bytes = create_test_image_bytes()
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    
    response = client.post("/analyze-frame", files=files)
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert "gender" in data["predictions"]
    print(f"✅ API test OK")

if __name__ == "__main__":
    print("="*60)
    print("API TESTS")
    print("="*60)
    test_root_endpoint()
    test_analyze_frame_success()
    print("\n✅ ALL API TESTS PASSED")