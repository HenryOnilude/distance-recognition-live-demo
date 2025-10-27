#!/usr/bin/env python3
"""
Debug Analysis Script
Run after testing to analyze debug output and diagnose issues
"""
import json
import os

def analyze_debug():
    print("=" * 60)
    print("DEBUG ANALYSIS")
    print("=" * 60)
    
    debug_dir = "debug_images"
    
    if not os.path.exists(debug_dir):
        print(f"❌ {debug_dir}/ directory not found!")
        return
    
    # Check for debug JSON
    json_file = f"{debug_dir}/debug_info.json"
    if os.path.exists(json_file):
        with open(json_file) as f:
            data = json.load(f)
        
        print("\n📊 DeepFace Results:")
        deepface = data['deepface_result']
        print(f"   Woman: {deepface['woman_prob']:.1f}%")
        print(f"   Man: {deepface['man_prob']:.1f}%")
        print(f"   Predicted: {deepface['predicted_gender']}")
        
        print("\n📊 Image Statistics:")
        stats = data['image_stats']
        print(f"   Mean BGR: {stats['mean_bgr']}")
        print(f"   Mean RGB: {stats['mean_rgb']}")
        
        # Check BGR→RGB conversion
        bgr_b = stats['mean_bgr'][0]
        rgb_b = stats['mean_rgb'][2]
        if abs(bgr_b - rgb_b) < 2.0:
            print(f"   ✅ BGR→RGB verified (BGR[0]={bgr_b:.1f} ≈ RGB[2]={rgb_b:.1f})")
        else:
            print(f"   ❌ Conversion FAILED!")

if __name__ == "__main__":
    analyze_debug()
    