"""Tests for all 15 MVTec AD categories, database product catalog, and Out-Of-Distribution (OOD) rejection."""

import pytest
import os
import sys
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from ml.inference.pipeline import InferencePipeline
from app.database.session import SessionLocal
from app.models.all_models import Product, ProductionBatch

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

@pytest.fixture(scope="module")
def pipeline():
    return InferencePipeline()

def test_all_15_categories_in_database_catalog():
    """Verify that all 15 MVTec categories are registered as products with batches in the database."""
    db = SessionLocal()
    try:
        products = db.query(Product).all()
        prod_names = [p.name.lower() for p in products]
        prod_codes = [p.product_code for p in products]
        
        for cat in CATEGORIES:
            matches = [name for name in prod_names if cat in name.replace("-", "_").replace(" ", "_")]
            assert len(matches) > 0, f"Category '{cat}' is missing from the Product database catalog!"
            
        batches = db.query(ProductionBatch).all()
        assert len(batches) >= 15, "At least 15 production batches must be registered!"
    finally:
        db.close()

def test_ood_explicit_unsupported_category_rejection(pipeline):
    """Verify that unsupported objects (e.g. smartphone, keyboard, mouse) are rejected with REVIEW decision."""
    unsupported_cases = [
        "Smartphone Device",
        "Mechanical Keyboard",
        "Computer Mouse",
        "Beverage Cup",
        "Household Furniture",
    ]
    test_img = "test_defect.png" if os.path.isfile("test_defect.png") else glob.glob("datasets/mvtec_raw/bottle/test/broken_large/*.png")[0]

    for item in unsupported_cases:
        result = pipeline.inspect_image(test_img, product_name=item)
        assert result["status"] == "unsupported_object", f"Failed to flag {item} as unsupported_object!"
        assert result["overall_decision"] == "REVIEW", f"Expected REVIEW decision for {item}, got {result['overall_decision']}"
        assert result["overall_level"] == "UNKNOWN"
        assert result["quality_assessment"]["manual_review_required"] is True
        assert "Unsupported" in result["quality_assessment"]["quality_risk"]

def test_normal_sample_returns_pass(pipeline):
    """Verify that a normal sample produces status='normal' and decision='PASS'."""
    normal_imgs = glob.glob("datasets/mvtec_raw/bottle/test/good/*.png")
    if normal_imgs:
        result = pipeline.inspect_image(normal_imgs[0], product_name="bottle")
        assert result["status"] == "normal"
        assert result["overall_decision"] == "PASS"
        assert len(result["defects"]) == 0

def test_defective_sample_returns_defect_and_classification(pipeline):
    """Verify that a defective sample is detected with a mapped defect classification."""
    defect_imgs = glob.glob("datasets/mvtec_raw/bottle/test/broken_large/*.png")
    if defect_imgs:
        result = pipeline.inspect_image(defect_imgs[0], product_name="bottle")
        assert result["status"] == "defective"
        assert len(result["defects"]) > 0
        assert result["overall_decision"] in ("FAIL", "REVIEW")
        top_defect = result["defects"][0]
        assert top_defect["product_category"] == "bottle"
        assert "broken" in top_defect["type"] or "large" in top_defect["type"]
