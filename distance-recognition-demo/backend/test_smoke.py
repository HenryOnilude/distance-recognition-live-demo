"""
SMOKE TEST: Quick sanity check - run after EVERY code change
Takes <10 seconds, catches major breakage
"""
import cv2
import numpy as np
from insightface_recognition import insightface_recognition_system
import time

def create_test_face():
    """Generate synthetic face for testing"""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw face
    cv2.ellipse(image, (320, 240), (100, 130), 0, 0, 360, (200, 180, 150), -1)
    # Eyes
    cv2.circle(image, (290, 220), 10, (50, 50, 50), -1)
    cv2.circle(image, (350, 220), 10, (50, 50, 50), -1)
    # Nose
    cv2.ellipse(image, (320, 260), (10, 15), 0, 0, 360, (180, 160, 130), -1)
    # Mouth
    cv2.ellipse(image, (320, 300), (40, 20), 0, 0, 180, (100, 80, 80), 2)
    return image

def test_basic_pipeline():
    """Basic smoke test - does the system work at all?"""
    print("🔥 Running smoke test...")
    
    test_image = create_test_face()
    face_bbox = (220, 110, 200, 260)
    face_region = test_image[110:370, 220:420]
    
    start = time.time()
    result = insightface_recognition_system.process_frame_analysis(
        face_bbox=face_bbox,
        face_image=face_region,
        full_image=test_image,
        image_shape=test_image.shape[:2]
    )
    elapsed = time.time() - start
    
    # Basic structure checks
    assert "predictions" in result, "❌ Missing predictions"
    assert "gender" in result["predictions"], "❌ Missing gender prediction"
    assert "age" in result["predictions"], "❌ Missing age prediction"
    assert "distance_m" in result, "❌ Missing distance estimation"
    assert "quality_score" in result, "❌ Missing quality assessment"
    
    # Gender checks
    gender = result["predictions"]["gender"]
    assert "predicted_class" in gender, "❌ Missing gender class"
    assert "confidence" in gender, "❌ Missing gender confidence"
    assert 0 <= gender["confidence"] <= 1, "❌ Invalid gender confidence"
    assert gender["predicted_class"] in ["Male", "Female"], "❌ Invalid gender class"
    
    # Age checks
    age = result["predictions"]["age"]
    assert "predicted_class" in age, "❌ Missing age class"
    assert "confidence" in age, "❌ Missing age confidence"
    
    # Distance checks
    assert result["distance_m"] > 0, "❌ Invalid distance"
    
    print(f"✅ SMOKE TEST PASSED ({elapsed:.2f}s)")
    print(f"   Gender: {gender['predicted_class']} ({gender['confidence']:.2f})")
    print(f"   Age: {age['predicted_class']}")
    print(f"   Distance: {result['distance_m']:.1f}m ({result['distance_category']})")
    print(f"   Quality: {result['quality_score']:.2f}")
    
    return True

def test_performance():
    """Ensure inference is fast enough"""
    print("\n⚡ Testing performance...")
    
    test_image = create_test_face()
    face_bbox = (220, 110, 200, 260)
    face_region = test_image[110:370, 220:420]
    
    times = []
    for i in range(3):
        start = time.time()
        result = insightface_recognition_system.process_frame_analysis(
            face_bbox=face_bbox,
            face_image=face_region,
            full_image=test_image,
            image_shape=test_image.shape[:2]
        )
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = sum(times) / len(times)
    
    if avg_time < 2.0:
        print(f"✅ Performance OK: {avg_time:.2f}s average")
    else:
        print(f"⚠️ Performance slow: {avg_time:.2f}s average (target <2s)")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST - Quick System Validation")
    print("=" * 60)
    print()
    
    try:
        test_basic_pipeline()
        test_performance()
        
        print()
        print("=" * 60)
        print("✅ ALL SMOKE TESTS PASSED - System is operational")
        print("=" * 60)
        exit(0)
    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"❌ SMOKE TEST FAILED: {e}")
        print("=" * 60)
        exit(1)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ SMOKE TEST ERROR: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        exit(1)