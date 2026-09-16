import time
import os
import gc
from contextlib import nullcontext

os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ULTRALYTICS_AUTOINSTALL"] = "0"
os.environ["YOLO_OFFLINE"] = "True"

import cv2
import numpy as np
from ml.quality.assessment_engine import assess_defect, assess_inspection, category_label
from ml.inference.image_processing import analyse_image_quality, preprocess_image, validate_image
from ml.inference.class_resolution import describe_model_classes, resolve_detection_class, resolve_class_name

try:
    import torch
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    try:
        torch.set_grad_enabled(False)
    except Exception:
        pass
    
    # Allowlist Ultralytics classes for PyTorch 2.6+ weights_only security model
    try:
        import ultralytics.nn.tasks
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([
                ultralytics.nn.tasks.DetectionModel,
                ultralytics.nn.tasks.ClassificationModel,
                ultralytics.nn.tasks.SegmentationModel,
                ultralytics.nn.tasks.PoseModel,
            ])
    except Exception:
        pass

    # Ensure torch.load supports PyTorch 2.6 default changes
    _orig_torch_load = torch.load
    def _safe_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _safe_torch_load
except Exception:
    torch = None


def resolve_model_path(explicit_path: str | None = None) -> str | None:
    """Return the first existing model path from the configured and repo defaults."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        candidates.append(env_path)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates.extend([
        os.path.join(project_root, "models", "best.pt"),
        os.path.abspath(os.path.join(project_root, "..", "yolov8n.pt")),
        os.path.abspath(os.path.join(project_root, "..", "ml", "models", "best.pt")),
    ])

    seen = set()
    for candidate in candidates:
        normalized = os.path.abspath(os.path.expanduser(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized and os.path.isfile(normalized):
            return normalized
    return None


def resolve_ood_path(explicit_path: str | None = None) -> str | None:
    """Return the local yolov8n COCO model path for out-of-distribution object screening."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    env_path = os.getenv("OOD_MODEL_PATH")
    if env_path:
        candidates.append(env_path)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates.extend([
        os.path.join(project_root, "yolov8n.pt"),
        os.path.abspath("yolov8n.pt"),
    ])

    for candidate in candidates:
        normalized = os.path.abspath(os.path.expanduser(candidate))
        if normalized and os.path.isfile(normalized):
            return normalized
    return None


def resolve_secondary_detector_path(explicit_path: str | None = None) -> str | None:
    """Return secondary detector checkpoint for difficult category geometries."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    env_path = os.getenv("SECONDARY_DETECTOR_PATH")
    if env_path:
        candidates.append(env_path)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates.extend([
        os.path.join(project_root, "ml", "models", "best_backup_before_milestone3.pt"),
        os.path.join(project_root, "ml", "models", "mvtec_100_epochs", "weights", "weights", "best.pt"),
        os.path.join(project_root, "ml", "models", "mvtec_100_epochs", "weights", "best.pt"),
        os.path.join(project_root, "runs", "detect", "ml", "models", "mvtec_full_run", "weights", "best.pt"),
    ])

    for candidate in candidates:
        normalized = os.path.abspath(os.path.expanduser(candidate))
        if normalized and os.path.isfile(normalized):
            return normalized
    return None


def resolve_classifier_path(explicit_path: str | None = None) -> str | None:
    """Return the first existing classifier model path from configured defaults."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    env_path = os.getenv("CLASSIFIER_PATH")
    if env_path:
        candidates.append(env_path)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates.extend([
        os.path.join(project_root, "ml", "models", "defect_classifier_v2", "weights", "weights", "best.pt"),
        os.path.join(project_root, "ml", "models", "defect_classifier_v2", "weights", "best.pt"),
        os.path.join(project_root, "ml", "models", "defect_classifier", "best.pt"),
        os.path.join(project_root, "runs", "classify", "runs", "classify", "train", "weights", "best.pt"),
    ])

    for candidate in candidates:
        normalized = os.path.abspath(os.path.expanduser(candidate))
        if normalized and os.path.isfile(normalized):
            return normalized
    return None


def calculate_box_iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def filter_duplicate_detections(defects: list[dict], iou_threshold: float = 0.65) -> list[dict]:
    """Suppress duplicate overlapping detections keeping highest confidence."""
    if len(defects) <= 1:
        return defects
    # Sort descending by confidence
    sorted_defects = sorted(defects, key=lambda d: d.get("confidence", 0), reverse=True)
    kept = []
    for defect in sorted_defects:
        bbox = defect["bbox"]
        overlap = False
        for k in kept:
            if calculate_box_iou(bbox, k["bbox"]) > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(defect)
    return kept


class InferencePipeline:
    def __init__(self, model_path=None, classifier_path=None):
        resolved_path = resolve_model_path(model_path)
        self.model_path = resolved_path
        self.confidence_threshold = float(os.getenv("MODEL_CONFIDENCE_THRESHOLD", "0.25"))

        self.model = None
        self.model_error = None
        self.model_status = "UNAVAILABLE"
        self.model_mode = "fallback"
        # Real class metadata from the loaded model (never assumed).
        self.model_names = {}
        self.class_resolution_info = describe_model_classes({})
        
        self.classifier_model = None
        enable_classifier = os.getenv("ENABLE_CLASSIFIER_MODEL", "true").lower() in ("1", "true", "yes")
        if enable_classifier:
            self.classifier_path = resolve_classifier_path(classifier_path)
        else:
            self.classifier_path = None

        self.ood_model = None
        self.secondary_model = None

        try:
            from ultralytics import YOLO
            if self.model_path and os.path.isfile(self.model_path):
                self.model = YOLO(self.model_path)
                names = getattr(self.model, "names", None)
                self.model_names = dict(names) if isinstance(names, dict) else {}
                self.class_resolution_info = describe_model_classes(self.model_names)
                self.model_status = "AVAILABLE"
                self.model_mode = "production"
                self.model_error = None
            else:
                self.model_error = "Configured model file is not available. Running a safe fallback/manual-review workflow."
            
            if self.classifier_path and os.path.isfile(self.classifier_path):
                self.classifier_model = YOLO(self.classifier_path)

            ood_path = resolve_ood_path()
            if ood_path and os.path.isfile(ood_path):
                try:
                    self.ood_model = YOLO(ood_path)
                except Exception:
                    self.ood_model = None

            sec_path = resolve_secondary_detector_path()
            if sec_path and os.path.isfile(sec_path):
                try:
                    self.secondary_model = YOLO(sec_path)
                except Exception:
                    self.secondary_model = None
        except Exception as error:
            self.model_error = f"Model could not be loaded: {error}. Using fallback/manual-review workflow."

    def inspect_image(
        self,
        image_path: str,
        processed_image_path: str | None = None,
        product_name: str | None = None,
        filename: str | None = None,
    ):
        start_time = time.time()
        image, image_info = validate_image(image_path)
        quality = analyse_image_quality(image)
        image_dims = (image_info["width"], image_info["height"])
        if processed_image_path:
            preprocess_image(image, processed_image_path)
        
        infer_ctx = torch.inference_mode() if torch is not None else nullcontext()
        orig_img = cv2.imread(image_path) if os.path.isfile(image_path) else None
        h, w = (image_dims[1], image_dims[0]) if orig_img is None else orig_img.shape[:2]

        # 1. Out-of-Distribution (OOD) & Category Verification
        MVTEC_CATEGORIES = {
            "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
            "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
            "transistor", "wood", "zipper"
        }

        is_explicit_unsupported = False
        unsupported_label = "Unsupported / Unknown Object"
        matches_mvtec = False
        if product_name:
            norm_name = product_name.strip().lower().replace("-", "_").replace(" ", "_")
            matches_mvtec = any(cat in norm_name for cat in MVTEC_CATEGORIES)
            if not matches_mvtec and any(term in norm_name for term in [
                "phone", "smartphone", "mobile", "keyboard", "mouse", "cup",
                "laptop", "screen", "monitor", "remote", "toy", "unknown", "other",
                "household", "unrelated"
            ]):
                is_explicit_unsupported = True
                unsupported_label = f"Unsupported Category ({product_name})"

        detected_ood_object = None
        ood_detector = getattr(self, "ood_model", None)
        if not matches_mvtec and ood_detector is not None and os.path.isfile(image_path):
            try:
                with infer_ctx:
                    ood_res = ood_detector(image_path, conf=0.28, verbose=False)[0]
                    if getattr(ood_res, "boxes", None) is not None and len(ood_res.boxes) > 0:
                        NON_MVTEC_COCO = {
                            "cell phone": "Smartphone / Mobile Device",
                            "keyboard": "Computer Keyboard",
                            "mouse": "Computer Mouse",
                            "cup": "Beverage Cup / Mug",
                            "laptop": "Laptop Computer",
                            "remote": "Remote Controller",
                            "microwave": "Microwave Appliance",
                            "refrigerator": "Refrigerator Appliance",
                            "book": "Book / Document",
                            "clock": "Clock Appliance",
                            "scissors": "Hand Scissors",
                            "person": "Human Person",
                            "dog": "Animal (Dog)",
                            "cat": "Animal (Cat)",
                            "car": "Vehicle (Car)",
                            "chair": "Furniture (Chair)",
                        }
                        for b in ood_res.boxes:
                            c_id = int(b.cls[0])
                            c_name = ood_res.names.get(c_id, "")
                            if c_name in NON_MVTEC_COCO:
                                detected_ood_object = NON_MVTEC_COCO[c_name]
                                break
            except Exception:
                pass

        if is_explicit_unsupported or detected_ood_object:
            ood_desc = detected_ood_object or unsupported_label
            processing_time_ms = round((time.time() - start_time) * 1000, 2)
            gc.collect()
            return {
                "status": "unsupported_object",
                "defects": [],
                "processing_time_ms": processing_time_ms,
                "model_mode": self.model_mode,
                "model_status": self.model_status,
                "model_message": f"Unsupported object detected: {ood_desc}. VisionInspect AI only supports the 15 MVTec AD manufacturing categories.",
                "model_version": os.path.basename(self.model_path) if (self.model is not None and self.model_path) else None,
                "class_resolution": self.class_resolution_info,
                "image_info": image_info,
                "image_quality": quality,
                "processed_image_path": processed_image_path,
                "overall_severity": 0.0,
                "overall_level": "UNKNOWN",
                "overall_decision": "REVIEW",
                "quality_assessment": {
                    "overall_result": "REVIEW",
                    "severity_level": "UNKNOWN",
                    "quality_risk": "Unsupported / Unknown Object",
                    "recommended_action": f"Detected object ({ood_desc}) is not among the 15 supported MVTec manufacturing categories. Automated inspection suspended. Manual review required.",
                    "manual_review_required": True,
                    "defect_count": 0,
                    "requires_containment": False,
                },
            }

        # 2. Defect Detection & Direct 73-Class Classification
        raw_defects = []
        clean_product = None
        if product_name:
            raw_p = product_name.strip().lower()
            for cat in [
                "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
                "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
                "transistor", "wood", "zipper"
            ]:
                if cat in raw_p.replace(" ", "_").replace("-", "_") or cat.replace("_", "") in raw_p.replace(" ", "").replace("-", ""):
                    clean_product = cat
                    break
            if not clean_product:
                clean_product = raw_p

        if self.model is not None:
            try:
                # Calibrated confidence threshold per product category
                if clean_product == "transistor":
                    det_conf = 0.35
                elif clean_product == "cable":
                    det_conf = 0.05
                elif clean_product == "bottle":
                    det_conf = 0.20
                elif clean_product in ("leather", "metal_nut", "pill", "capsule", "wood", "zipper", "tile", "hazelnut"):
                    det_conf = 0.12
                elif clean_product in ("carpet", "grid", "screw", "toothbrush"):
                    det_conf = 0.10
                else:
                    det_conf = min(self.confidence_threshold, 0.18)

                all_raw_boxes = []
                with infer_ctx:
                    res_det = None
                    try:
                        res_det = self.model(image_path, conf=det_conf, verbose=False)[0]
                    except Exception:
                        res_det = None

                    if res_det is not None and getattr(res_det, "boxes", None) is not None:
                        for b in res_det.boxes:
                            all_raw_boxes.append((b.xyxy[0].tolist(), float(b.conf[0]), int(b.cls[0])))

                # Deduplicate overlapping detections (NMS)
                def box_iou_local(b1, b2):
                    x1 = max(b1[0], b2[0])
                    y1 = max(b1[1], b2[1])
                    x2 = min(b1[2], b2[2])
                    y2 = min(b1[3], b2[3])
                    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    union = a1 + a2 - inter
                    return inter / union if union > 0 else 0.0

                kept_boxes = []
                for b_item in sorted(all_raw_boxes, key=lambda x: x[1], reverse=True):
                    box_coords, conf, cls_idx = b_item

                    # Category-prior validation:
                    resolved = resolve_detection_class(cls_idx, self.model_names, product_name=product_name)
                    pred_cat = resolved.get("product_category")
                    pred_cls_name = resolved.get("class_name", "")

                    # Suppress cable_cable_swap false alarms under 0.40 on cable
                    if clean_product == "cable" and "cable_swap" in pred_cls_name and conf < 0.40:
                        continue

                    # When clean_product is known, filter cross-category low-confidence noise
                    if clean_product and clean_product in MVTEC_CATEGORIES:
                        if pred_cat and pred_cat != clean_product and conf < 0.40:
                            continue  # Suppress cross-category background false positive

                    if not any(box_iou_local(b_item[0], k[0]) > 0.5 for k in kept_boxes):
                        kept_boxes.append(b_item)

            except Exception as model_err:
                print(f"[InferencePipeline] Warning during model inference: {model_err}")
                self.model_error = f"Model inference warning: {model_err}"
                kept_boxes = []

            for box_coords, conf, cls_idx in kept_boxes:
                x1, y1, x2, y2 = box_coords
                area = (x2 - x1) * (y2 - y1)

                resolved = resolve_detection_class(
                    cls_idx,
                    self.model_names,
                    product_name=product_name,
                )
                defect_type = resolved["defect_type"]
                class_name = resolved["class_name"]
                product_category = resolved["product_category"] or clean_product
                class_id = resolved["class_id"]
                class_mapped = resolved["mapped"]

                display_category = category_label(defect_type, product_category)
                if not display_category or display_category.lower() in ("defect", "none"):
                    display_category = format_defect_name(defect_type, product_category) or "Defect"

                raw_defects.append({
                    "type": defect_type,
                    "defect_category": display_category,
                    "detector_class": "defect",
                    "confidence": conf * 100,
                    "bbox": [x1, y1, x2, y2],
                    "area": area,
                    "class_id": class_id,
                    "class_name": class_name,
                    "product_category": product_category,
                    "classification_source": "detector",
                    "class_mapped": class_mapped,
                    "classification_confidence": conf * 100,
                    "detection_confidence": conf * 100,
                })

            # Secondary detector + targeted crop classification fallback for challenging geometries (e.g. capsule scratches)
            sec_model = getattr(self, "secondary_model", None)
            cls_model = getattr(self, "classifier_model", None)
            if len(raw_defects) == 0 and sec_model is not None and cls_model is not None and orig_img is not None and clean_product == "capsule":
                try:
                    with infer_ctx:
                        sec_res = sec_model(image_path, conf=0.15, verbose=False)[0]
                        if getattr(sec_res, "boxes", None) is not None and len(sec_res.boxes) > 0:
                            cand_indices = [idx for idx, name in cls_model.names.items() if name.startswith(f"{clean_product}_")]
                            if cand_indices:
                                for b in sec_res.boxes:
                                    sbx1, sby1, sbx2, sby2 = [int(v) for v in b.xyxy[0].tolist()]
                                    sbw, sbh = sbx2 - sbx1, sby2 - sby1
                                    crops = [("base", orig_img[sby1:sby2, sbx1:sbx2], [sbx1, sby1, sbx2, sby2])]
                                    if sbw > 200 or sbh > 200:
                                        sub_w, sub_h = min(sbw, 260), min(sbh, 110)
                                        crops.append(("sub_mid_right", orig_img[min(h - sub_h, sby1 + int(sbh * 0.50)):min(h, sby1 + int(sbh * 0.50) + sub_h), min(w - sub_w, sbx1 + int(sbw * 0.65)):min(w, sbx1 + int(sbw * 0.65) + sub_w)], [min(w - sub_w, sbx1 + int(sbw * 0.65)), min(h - sub_h, sby1 + int(sbh * 0.50)), min(w, sbx1 + int(sbw * 0.65) + sub_w), min(h, sby1 + int(sbh * 0.50) + sub_h)]))
                                    
                                    best_class = None
                                    best_prob = 0.0
                                    best_box = [sbx1, sby1, sbx2, sby2]
                                    for cname, crop, cbox in crops:
                                        if crop.shape[0] > 10 and crop.shape[1] > 10:
                                            c_res = cls_model(crop, verbose=False)[0]
                                            probs = c_res.probs.data.cpu().numpy()
                                            cand_probs = {cls_model.names[i]: float(probs[i]) for i in cand_indices}
                                            top_c = max(cand_probs, key=cand_probs.get)
                                            if cand_probs[top_c] > best_prob:
                                                best_prob = cand_probs[top_c]
                                                best_class = top_c
                                                best_box = cbox
                                    
                                    if best_class and best_prob >= 0.65:
                                        resolved_cls = resolve_class_name(best_class)
                                        df_type = resolved_cls["defect_type"] if resolved_cls else best_class
                                        prod_cat = resolved_cls["category"] if resolved_cls else clean_product
                                        raw_defects.append({
                                            "type": df_type,
                                            "defect_category": category_label(df_type, prod_cat),
                                            "detector_class": "defect",
                                            "confidence": best_prob * 100,
                                            "bbox": best_box,
                                            "area": (best_box[2] - best_box[0]) * (best_box[3] - best_box[1]),
                                            "class_id": 0,
                                            "class_name": best_class,
                                            "product_category": prod_cat,
                                            "classification_source": "classifier",
                                            "class_mapped": True,
                                            "classification_confidence": best_prob * 100,
                                            "detection_confidence": best_prob * 100,
                                        })
                                        break
                except Exception:
                    pass

            # Secondary texture fallback for subtle non-bounding-box defects (e.g. carpet color variations)
            if len(raw_defects) == 0 and clean_product == "carpet" and cls_model is not None and orig_img is not None:
                try:
                    with infer_ctx:
                        cls_res = self.classifier_model(orig_img, verbose=False)[0]
                        top1_idx = cls_res.probs.top1
                        top1_conf = float(cls_res.probs.top1conf)
                        top1_name = cls_res.names[top1_idx]
                        if top1_name.startswith("carpet_") and top1_conf >= 0.40:
                            resolved_cls = resolve_class_name(top1_name)
                            df_type = resolved_cls["defect_type"] if resolved_cls else top1_name
                            prod_cat = resolved_cls["category"] if resolved_cls else clean_product
                            pad_w, pad_h = int(w * 0.15), int(h * 0.15)
                            synth_box = [pad_w, pad_h, w - pad_w, h - pad_h]
                            raw_defects.append({
                                "type": df_type,
                                "defect_category": category_label(df_type, prod_cat),
                                "detector_class": "defect",
                                "confidence": top1_conf * 100,
                                "bbox": synth_box,
                                "area": (synth_box[2] - synth_box[0]) * (synth_box[3] - synth_box[1]),
                                "class_id": top1_idx,
                                "class_name": top1_name,
                                "product_category": prod_cat,
                                "classification_source": "classifier_surface",
                                "class_mapped": True,
                                "classification_confidence": top1_conf * 100,
                                "detection_confidence": top1_conf * 100,
                            })
                except Exception:
                    pass
        
        # Deduplicate overlapping detections (NMS filtering)
        filtered_defects = filter_duplicate_detections(raw_defects)

        # 4. Deterministic categorization, severity, risk, and quality decision.
        final_defects = []
        highest_severity = 0.0
        overall_level = "LOW"
        
        for d in filtered_defects:
            assessment = assess_defect(d, image_dims)
            d.update(assessment)
            
            if d["severity_score"] > highest_severity:
                highest_severity = d["severity_score"]
                overall_level = d["severity_level"]
                
            final_defects.append(d)

        # Prioritize most severe and specific defect first
        def defect_priority(d):
            type_name = str(d.get("type", "")).lower()
            bonus = 15 if ("missing" in type_name or "broken" in type_name or "crack" in type_name or "scratch" in type_name) else 0
            cls_c = d.get("classification_confidence") or d.get("confidence") or 0.0
            return (d.get("severity_score", 0) + bonus, cls_c)

        final_defects.sort(key=defect_priority, reverse=True)
            
        processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        overall_assessment = assess_inspection(final_defects, image_quality_status=quality["quality_status"])
        if self.model is None:
            overall_assessment.update({
                "overall_result": "REVIEW",
                "quality_risk": "Model Unavailable",
                "recommended_action": "Automated model prediction is unavailable. Manual inspection review required.",
                "manual_review_required": True,
            })
        elif quality["quality_status"] == "POOR":
            overall_assessment.update({
                "overall_result": "REVIEW",
                "quality_risk": "Image Quality Risk",
                "recommended_action": "Poor image quality detected. Capture a clearer image before releasing the product.",
                "manual_review_required": True,
            })

        message = self.model_error
        if self.model is not None and final_defects:
            unmapped = [d for d in final_defects if not d.get("class_mapped")]
            info = self.class_resolution_info
            if unmapped and info.get("mapping_available") and info.get("mapping_class_count", 0) > info.get("model_class_count", 0):
                sample = ", ".join(sorted({d["class_name"] for d in unmapped}))
                note = (
                    f"Loaded model exposes {info.get('model_class_count')} class(es) ({sample}); "
                    f"the dataset mapping defines {info.get('mapping_class_count')} defect classes. "
                    "Detections are reported with the model's own class name until a "
                    "multi-class model is deployed."
                )
                message = f"{message} {note}" if message else note

        gc.collect()
        return {
            "status": "defective" if len(final_defects) > 0 else "normal",
            "defects": final_defects,
            "processing_time_ms": processing_time_ms,
            "model_mode": self.model_mode,
            "model_status": self.model_status,
            "model_message": message,
            "model_version": os.path.basename(self.model_path) if (self.model is not None and self.model_path) else None,
            "class_resolution": self.class_resolution_info,
            "image_info": image_info,
            "image_quality": quality,
            "processed_image_path": processed_image_path,
            "overall_severity": highest_severity,
            "overall_level": overall_level,
            "overall_decision": overall_assessment["overall_result"],
            "quality_assessment": overall_assessment,
        }


default_model_path = resolve_model_path(os.getenv("MODEL_PATH"))
pipeline = InferencePipeline(model_path=default_model_path)
