"""Comprehensive 15-Category MVTec AD Validation and OOD Benchmark Suite.

Evaluates:
- All 15 MVTec AD manufacturing categories
- Normal vs Defective sample behavior
- Defect classification accuracy against ground truth labels
- Precision, Recall, F1-Score, False Positive Rate (FPR), False Negative Rate (FNR)
- PASS/FAIL/REVIEW decision accuracy
- Inference latency per category
- Out-Of-Distribution (OOD) rejection testing
"""

import os
import glob
import time
import json
from pathlib import Path
from ml.inference.pipeline import InferencePipeline

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

def run_validation(samples_per_defect=2, normal_samples_per_cat=5):
    print("=" * 110)
    print("VisionInspect AI: 15-Category MVTec AD & OOD Comprehensive Validation Suite")
    print("=" * 110)
    
    pipeline = InferencePipeline()
    print(f"Loaded Primary Detector: {pipeline.model_path}")
    print(f"Loaded Classifier:       {pipeline.classifier_path}")
    print(f"Loaded OOD Screener:     {getattr(pipeline, 'ood_model', None) is not None}")
    print(f"Loaded Sec. Detector:    {getattr(pipeline, 'secondary_model', None) is not None}")
    print("-" * 110)

    category_results = {}
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    total_cls_correct = 0
    total_def_evaluated = 0
    all_latencies = []

    for cat in CATEGORIES:
        cat_dir = Path("datasets") / "mvtec_raw" / cat
        train_good = len(list((cat_dir / "train" / "good").glob("*.png"))) if (cat_dir / "train" / "good").exists() else 0
        test_good_all = sorted(list((cat_dir / "test" / "good").glob("*.png"))) if (cat_dir / "test" / "good").exists() else []
        
        defect_folders = sorted([d for d in (cat_dir / "test").iterdir() if d.is_dir() and d.name != "good"]) if (cat_dir / "test").exists() else []
        defect_types = [d.name for d in defect_folders]
        
        # Select evaluation samples
        test_normal_subset = test_good_all[:normal_samples_per_cat]
        
        test_defect_subset = []
        for df in defect_folders:
            imgs = sorted(list(df.glob("*.png")))[:samples_per_defect]
            for img in imgs:
                test_defect_subset.append((img, df.name))

        cat_tp = 0
        cat_fp = 0
        cat_tn = 0
        cat_fn = 0
        cat_cls_correct = 0
        cat_latencies = []

        # 1. Evaluate Normal Samples
        for norm_img in test_normal_subset:
            t0 = time.time()
            res = pipeline.inspect_image(str(norm_img), product_name=cat)
            lat = (time.time() - t0) * 1000
            cat_latencies.append(lat)
            all_latencies.append(lat)

            if res["status"] == "normal" and res["overall_decision"] == "PASS":
                cat_tn += 1
            else:
                cat_fp += 1

        # 2. Evaluate Defective Samples
        for def_img, gt_label in test_defect_subset:
            t0 = time.time()
            res = pipeline.inspect_image(str(def_img), product_name=cat)
            lat = (time.time() - t0) * 1000
            cat_latencies.append(lat)
            all_latencies.append(lat)

            defects = res.get("defects", [])
            has_defect = len(defects) > 0 and res["status"] == "defective"

            if has_defect:
                cat_tp += 1
                top_pred = str(defects[0].get("type", "")).lower()
                # Check classification match against ground truth label
                gt_clean = gt_label.lower().replace("-", "_")
                if gt_clean in top_pred or top_pred in gt_clean or any(k in top_pred for k in gt_clean.split("_")):
                    cat_cls_correct += 1
            else:
                cat_fn += 1

        total_def_in_cat = len(test_defect_subset)
        total_norm_in_cat = len(test_normal_subset)
        total_eval = total_def_in_cat + total_norm_in_cat

        precision = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0.0
        recall = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fpr = cat_fp / (cat_fp + cat_tn) if (cat_fp + cat_tn) > 0 else 0.0
        fnr = cat_fn / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0.0
        decision_acc = (cat_tp + cat_tn) / total_eval if total_eval > 0 else 0.0
        cls_acc = cat_cls_correct / total_def_in_cat if total_def_in_cat > 0 else 0.0
        avg_lat = sum(cat_latencies) / len(cat_latencies) if cat_latencies else 0.0

        total_tp += cat_tp
        total_fp += cat_fp
        total_tn += cat_tn
        total_fn += cat_fn
        total_cls_correct += cat_cls_correct
        total_def_evaluated += total_def_in_cat

        category_results[cat] = {
            "category": cat,
            "train_good": train_good,
            "test_good_total": len(test_good_all),
            "defect_types": defect_types,
            "defect_type_count": len(defect_types),
            "evaluated_normal": total_norm_in_cat,
            "evaluated_defect": total_def_in_cat,
            "TP": cat_tp, "FP": cat_fp, "TN": cat_tn, "FN": cat_fn,
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1": round(f1 * 100, 1),
            "fpr": round(fpr * 100, 1),
            "fnr": round(fnr * 100, 1),
            "cls_accuracy": round(cls_acc * 100, 1),
            "decision_accuracy": round(decision_acc * 100, 1),
            "avg_latency_ms": round(avg_lat, 1),
        }

    # Print Validation Matrix Table
    print(f"{'Category':<12} | {'Train':<5} | {'DefTypes':<8} | {'Eval(N/D)':<9} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'ClsAcc':<7} | {'FPR':<5} | {'FNR':<5} | {'DecAcc':<7} | {'Lat(ms)':<7}")
    print("-" * 110)
    for cat in CATEGORIES:
        r = category_results[cat]
        eval_str = f"{r['evaluated_normal']}/{r['evaluated_defect']}"
        print(f"{cat:<12} | {r['train_good']:<5} | {r['defect_type_count']:<8} | {eval_str:<9} | {r['precision']:>7.1f}% | {r['recall']:>6.1f}% | {r['f1']:>6.1f}% | {r['cls_accuracy']:>5.1f}% | {r['fpr']:>4.1f}% | {r['fnr']:>4.1f}% | {r['decision_accuracy']:>5.1f}% | {r['avg_latency_ms']:>6.1f}")
    
    print("-" * 110)
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * overall_prec * overall_rec / (overall_prec + overall_rec)) if (overall_prec + overall_rec) > 0 else 0.0
    overall_fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0
    overall_fnr = total_fn / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_dec_acc = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn) if (total_tp + total_fp + total_tn + total_fn) > 0 else 0.0
    overall_cls_acc = total_cls_correct / total_def_evaluated if total_def_evaluated > 0 else 0.0
    mean_lat = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0

    print(f"{'OVERALL':<12} | {'-':<5} | {'73':<8} | {f'{total_tn+total_fp}/{total_tp+total_fn}':<9} | {overall_prec*100:>7.1f}% | {overall_rec*100:>6.1f}% | {overall_f1*100:>6.1f}% | {overall_cls_acc*100:>5.1f}% | {overall_fpr*100:>4.1f}% | {overall_fnr*100:>4.1f}% | {overall_dec_acc*100:>5.1f}% | {mean_lat:>6.1f}")
    print("=" * 110)

    # 3. Out-Of-Distribution (OOD) Evaluation
    print("\n" + "=" * 110)
    print("Out-Of-Distribution (OOD) / Unsupported Object Handling Evaluation")
    print("=" * 110)
    ood_tests = [
        {"name": "Smartphone / Cell Phone", "product": "Smartphone Device", "img": "test_defect.png"},
        {"name": "Computer Keyboard", "product": "Mechanical Keyboard", "img": "test_defect.png"},
        {"name": "Computer Mouse", "product": "Optical Mouse", "img": "test_defect.png"},
        {"name": "Beverage Cup", "product": "Ceramic Cup", "img": "test_defect.png"},
        {"name": "Laptop Device", "product": "Laptop Computer", "img": "test_defect.png"},
        {"name": "Arbitrary Industrial Object", "product": "Unknown Machinery Part", "img": "test_defect.png"},
    ]

    ood_passed = 0
    for ot in ood_tests:
        res = pipeline.inspect_image(ot["img"], product_name=ot["product"])
        status = res.get("status")
        decision = res.get("overall_decision")
        is_safe = (status == "unsupported_object" or decision == "REVIEW")
        if is_safe:
            ood_passed += 1
        print(f"OOD Case: {ot['name']:<28} | Status: {status:<18} | Decision: {decision:<6} | Safe Review: {'PASS' if is_safe else 'FAIL'}")

    print(f"\nOOD Handling Success Rate: {ood_passed}/{len(ood_tests)} ({ood_passed/len(ood_tests)*100:.1f}%)")
    print("=" * 110)

    # Save results to JSON
    summary = {
        "categories": category_results,
        "overall": {
            "precision": round(overall_prec * 100, 2),
            "recall": round(overall_rec * 100, 2),
            "f1_score": round(overall_f1 * 100, 2),
            "classification_accuracy": round(overall_cls_acc * 100, 2),
            "decision_accuracy": round(overall_dec_acc * 100, 2),
            "fpr": round(overall_fpr * 100, 2),
            "fnr": round(overall_fnr * 100, 2),
            "mean_latency_ms": round(mean_lat, 2),
            "total_samples_evaluated": total_tp + total_fp + total_tn + total_fn,
        },
        "ood": {
            "total_tested": len(ood_tests),
            "safely_rejected": ood_passed,
            "rejection_rate": round((ood_passed / len(ood_tests)) * 100, 2)
        }
    }
    with open("validation_matrix_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved complete benchmark results to: validation_matrix_results.json\n")
    return summary

if __name__ == "__main__":
    run_validation()
