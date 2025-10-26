"""
Test data types and formats throughout the pipeline
Catches type errors like the numpy.float32 bug
"""
import numpy as np
import cv2
from insightface_recognition import insightface_recognition_system

def test_numpy_types_accepted():
    """REGRESSION: numpy.float32 should be accepted"""
    print("Testing numpy type handling...")
    
    # This was your bug - should NOT raise TypeError
    gender_score = np.float32(0.75)
    confidence = np.float32(0.85)
    
    try:
        result = insightface_recognition_system.process_gender_prediction(
            gender_score, confidence, "close", 0.8
        )
        assert result['confidence'] > 0, "❌ Invalid confidence"
        print("✅ numpy.float32 handled correctly")
    except TypeError as e:
        print(f"❌ FAILED: {e}")
        raise

def test_bgr_image_format():
    """Verify preprocessing functions handle BGR correctly"""
    print("Testing BGR image format...")
    
    bgr_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # These should all work with BGR
    from quality_assessment import assess_image_quality
    from adaptive_preprocessing import preprocess_for_distance
    from distance_estimation import estimate_distance_from_face_size
    
    # Quality assessment
    quality = assess_image_quality(bgr_image)
    assert 0 <= quality <= 1, f"❌ Quality out of range: {quality}"
    
    # Preprocessing
    preprocessed = preprocess_for_distance(bgr_image, 3.0, 0.7)
    assert preprocessed.shape == bgr_image.shape, "❌ Preprocessing changed shape"
    
    # Distance estimation
    distance = estimate_distance_from_face_size((100, 100, 200, 250), bgr_image.shape[:2])
    assert 0.5 <= distance <= 15.0, f"❌ Distance out of range: {distance}"
    
    print("✅ BGR format handled correctly")

def test_api_response_structure():
    """Verify response has all required fields"""
    print("Testing API response structure...")
    
    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.ellipse(test_image, (320, 240), (100, 130), 0, 0, 360, (200, 180, 150), -1)
    
    face_bbox = (220, 110, 200, 260)
    face_region = test_image[110:370, 220:420]
    
    result = insightface_recognition_system.process_frame_analysis(
        face_bbox=face_bbox,
        face_image=face_region,
        full_image=test_image,
        image_shape=test_image.shape[:2]
    )
    
    # Check required fields
    required_fields = ["predictions", "distance_m", "distance_category", "quality_score", "processing_time_ms"]
    for field in required_fields:
        assert field in result, f"❌ Missing required field: {field}"
    
    # Check predictions structure
    assert "gender" in result["predictions"], "❌ Missing predictions.gender"
    assert "age" in result["predictions"], "❌ Missing predictions.age"
    
    # Check gender structure
    gender = result["predictions"]["gender"]
    assert "predicted_class" in gender, "❌ Missing gender.predicted_class"
    assert "confidence" in gender, "❌ Missing gender.confidence"
    
    # Check age structure
    age = result["predictions"]["age"]
    assert "predicted_class" in age, "❌ Missing age.predicted_class"
    assert "confidence" in age, "❌ Missing age.confidence"
    
    # Check distance structure (at root level)
    assert result["distance_m"] > 0, "❌ Invalid distance_m"
    assert result["distance_category"] in ["portrait", "close", "medium", "far"], "❌ Invalid distance_category"
    
    print("✅ API response structure valid")

def test_edge_case_inputs():
    """Test edge cases that might cause crashes"""
    print("Testing edge cases...")
    
    from distance_estimation import estimate_distance_from_face_size
    
    # Zero width face
    distance = estimate_distance_from_face_size((100, 100, 0, 100), (480, 640))
    assert distance > 0, "❌ Zero width face should return fallback distance"
    
    # Negative dimensions (shouldn't happen, but handle gracefully)
    distance = estimate_distance_from_face_size((100, 100, -10, 100), (480, 640))
    assert distance > 0, "❌ Negative width should return fallback distance"
    
    # Tiny face
    distance = estimate_distance_from_face_size((100, 100, 10, 10), (480, 640))
    assert distance >= 0.5, "❌ Tiny face distance out of range"
    
    # Huge face
    distance = estimate_distance_from_face_size((100, 100, 500, 600), (480, 640))
    assert distance >= 0.5, "❌ Huge face distance out of range"
    
    print("✅ Edge cases handled correctly")

if __name__ == "__main__":
    print("=" * 60)
    print("DATA FORMAT VALIDATION TESTS")
    print("=" * 60)
    print()
    
    try:
        test_numpy_types_accepted()
        print()
        test_bgr_image_format()
        print()
        test_api_response_structure()
        print()
        test_edge_case_inputs()
        
        print()
        print("=" * 60)
        print("✅ ALL DATA FORMAT TESTS PASSED")
        print("=" * 60)
        exit(0)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        exit(1)