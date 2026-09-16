import os
import sys

# Ensure project paths
sys.path.insert(0, os.path.abspath('.'))

from ml.inference.pipeline import pipeline

test_cases = [
    ("hazelnut", "crack", "datasets/mvtec_raw/hazelnut/test/crack/000.png", "Crack"),
    ("bottle", "broken_large", "datasets/mvtec_raw/bottle/test/broken_large/000.png", "Broken Large"),
    ("cable", "missing_cable", "datasets/mvtec_raw/cable/test/missing_cable/000.png", "Missing Cable"),
    ("capsule", "scratch", "datasets/mvtec_raw/capsule/test/scratch/000.png", "Scratch"),
]

print("="*80)
print("VERIFYING INFERENCE PIPELINE ON 4 REAL MVTEC TEST IMAGES")
print("="*80)

all_passed = True
for prod, defect_exp, path, expected_cat in test_cases:
    res = pipeline.inspect_image(path, product_name=prod)
    top_defect = res["defects"][0] if res["defects"] else None
    actual_cat = top_defect["defect_category"] if top_defect else "None"
    matched = (actual_cat == expected_cat)
    status_icon = "PASS" if matched else "FAIL"
    print(f"[{status_icon}] {prod:<10} {defect_exp:<15} -> Expected: '{expected_cat:<15}', Actual: '{actual_cat:<15}' (Decision: {res['overall_decision']})")
    if top_defect:
        print(f"       Class: {top_defect['class_name']}, Conf: {top_defect['confidence']:.1f}%, Cls Conf: {top_defect['classification_confidence']:.1f}%")
    if not matched:
        all_passed = False

print("\n" + "="*80)
print("FINAL PIPELINE VERIFICATION RESULT:", "SUCCESS (ALL 4 MATCHED)" if all_passed else "FAILURE")
print("="*80)
