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

import cv2
try:
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

try:
    import torch
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    
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

    _orig_torch_load = torch.load
    def _safe_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _safe_torch_load
except Exception:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, products, batches, inspections, analytics, models, reports
from app.database.session import engine
from app.models.all_models import Base

from fastapi.staticfiles import StaticFiles

from app.core.config import settings

app = FastAPI(title="VISIONINSPECT AI", version="1.0.0")

# Ensure uploads directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

cors_origins_env = os.getenv("CORS_ORIGINS", "").strip()
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://vision-ai-inspect-frontend-prod.onrender.com",
    "https://vision-ai-inspect-frontend.onrender.com",
    "https://visioninspect-frontend.onrender.com",
    "https://vision-ai-inspect.onrender.com",
]
if cors_origins_env and cors_origins_env != "*":
    for origin in cors_origins_env.split(","):
        cleaned = origin.strip().rstrip("/")
        if cleaned and cleaned not in allowed_origins:
            allowed_origins.append(cleaned)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount both /api and /api/v1 router prefixes for complete backward/forward compatibility
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(products.router, prefix="/api/products", tags=["products"])
app.include_router(products.router, prefix="/api/v1/products", tags=["products"])
app.include_router(batches.router, prefix="/api/batches", tags=["batches"])
app.include_router(batches.router, prefix="/api/v1/batches", tags=["batches"])
app.include_router(inspections.router, prefix="/api/inspections", tags=["inspections"])
app.include_router(inspections.router, prefix="/api/v1/inspections", tags=["inspections"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["reports"])

@app.get("/health")
def health_check():
    from ml.inference.pipeline import pipeline
    return {
        "status": "ok",
        "version": "v1.2.0-single-model",
        "model_status": pipeline.model_status,
        "model_path": pipeline.model_path,
        "classifier_path": pipeline.classifier_path,
        "classifier_loaded": pipeline.classifier_model is not None,
        "classifier_file_exists": os.path.isfile(pipeline.classifier_path) if pipeline.classifier_path else False,
    }


@app.on_event("startup")
def create_tables_on_startup():
    """Ensure database tables, default roles, and 15 MVTec catalog products exist when the app starts."""
    try:
        Base.metadata.create_all(bind=engine)
        from app.database.session import SessionLocal
        from app.models.all_models import Role, User, Product, ProductionBatch
        from app.core.security import get_password_hash
        
        db = SessionLocal()
        try:
            roles = ["ADMIN", "QUALITY_ENGINEER", "SUPERVISOR", "OPERATOR"]
            for role_name in roles:
                if not db.query(Role).filter(Role.name == role_name).first():
                    db.add(Role(name=role_name))
            db.commit()

            roles_map = {r.name: r for r in db.query(Role).all()}
            seed_accounts = [
                {"username": "admin", "email": "admin@visioninspect.local", "role": "ADMIN", "password": "admin123"},
                {"username": "quality_eng", "email": "quality_eng@visioninspect.ai", "role": "QUALITY_ENGINEER", "password": "quality123"},
                {"username": "Quality_Engineer", "email": "qualityengineer@gmail.com", "role": "QUALITY_ENGINEER", "password": "quality123"},
                {"username": "qualityengineer", "email": "qualityengineer1@gmail.com", "role": "QUALITY_ENGINEER", "password": "quality123"},
                {"username": "Demo", "email": "demo_final@gmail.com", "role": "QUALITY_ENGINEER", "password": "quality123"},
                {"username": "vicky", "email": "vicky22@gmail.com", "role": "SUPERVISOR", "password": "vicky123"},
                {"username": "vicky12", "email": "supervisor12@test.com", "role": "SUPERVISOR", "password": "vicky123"},
                {"username": "supervisor", "email": "supervisor@visioninspect.ai", "role": "SUPERVISOR", "password": "supervisor123"},
                {"username": "ramya", "email": "ramya@visioninspect.ai", "role": "QUALITY_ENGINEER", "password": "ramya123"},
            ]
            for sa in seed_accounts:
                target_role = roles_map.get(sa["role"])
                if not target_role:
                    continue
                user_record = db.query(User).filter(
                    (User.username == sa["username"]) | (User.email == sa["email"])
                ).first()
                if not user_record:
                    user_record = User(
                        username=sa["username"],
                        email=sa["email"],
                        hashed_password=get_password_hash(sa["password"]),
                        role_id=target_role.id
                    )
                    db.add(user_record)
                else:
                    user_record.hashed_password = get_password_hash(sa["password"])
                    user_record.role_id = target_role.id
                    user_record.is_active = True
            db.commit()

            # Seed all 15 MVTec product categories and standard production lines
            mvtec_catalog = [
                ("Bottle Container Inspection", "BTL-001", "Line 1 - Bottling", "Translucent glass & PET container defect detection"),
                ("Cable Wiring Harness Assembly", "CBL-002", "Line 2 - Wire Harness", "Multi-core insulated wiring cables and connectors"),
                ("Capsule Pharmaceutical Solid Dose", "CAP-003", "Line 3 - Packaging", "Hard gelatin and vegetable capsules"),
                ("Woven Carpet Textile Surface", "CPT-004", "Line 4 - Textiles", "Tufted and woven industrial carpets"),
                ("Metallic Mesh Grid Structure", "GRD-005", "Line 5 - Fabrication", "Precision woven metallic grids and filters"),
                ("Hazelnut Organic Product Sorting", "HZN-006", "Line 6 - Sorting", "Shelled and whole industrial food nuts"),
                ("Finished Leather Surface Material", "LTH-007", "Line 7 - Tannery", "Tanned upholstery and automotive leather sheets"),
                ("Hexagonal Metal Nut Fastener", "NUT-008", "Line 8 - Hardware", "Machined steel and brass threaded nuts"),
                ("Pharmaceutical Compressed Tablet Pill", "PIL-009", "Line 9 - Pharma", "Coated medical tablets and pills"),
                ("Industrial Threaded Steel Screw", "SCR-010", "Line 10 - Fasteners", "Countersunk and pan-head threaded machine screws"),
                ("Glazed Ceramic Floor Tile", "TIL-011", "Line 11 - Ceramics", "Polished and textured architectural ceramic tiles"),
                ("Molded Bristle Toothbrush Head", "TBH-012", "Line 12 - Consumer", "Hygiene brush heads and embedded bristle bundles"),
                ("Electronic Semiconductor Transistor", "TRS-013", "Line 13 - Electronics", "Through-hole TO-92 and power semiconductor packages"),
                ("Natural Wood Floor Plank Surface", "WOD-014", "Line 14 - Lumber", "Milled timber and hardwood floorboards"),
                ("Molded Coil Zipper Fastener Chain", "ZIP-015", "Line 15 - Apparel", "Continuous coil and tooth apparel zippers"),
                ("PCB Logic Board Assembly", "PCB-X100", "Line 16 - SMT", "High-density printed circuit boards with surface mount components"),
                ("Precision Gearbox Transmission", "GR-204", "Line 17 - Machining", "Precision machined automotive gears and bearings"),
                ("Other", "OTH-999", "Line 99 - General", "Custom objects, household items, or user-defined items with automated OOD screening"),
            ]

            for name, code, line, desc in mvtec_catalog:
                product = db.query(Product).filter(Product.product_code == code).first()
                if not product:
                    product = Product(name=name, product_code=code, production_line=line, description=desc)
                    db.add(product)
                    db.commit()
                    db.refresh(product)
                
                # Ensure default batch exists for each product
                batch_num = f"BATCH-{code}"
                batch = db.query(ProductionBatch).filter(ProductionBatch.batch_number == batch_num).first()
                if not batch:
                    batch = ProductionBatch(batch_number=batch_num, product_id=product.id)
                    db.add(batch)
                    db.commit()
        finally:
            db.close()

        # Warmup model inference at startup to eliminate cold-start latency on first user request
        try:
            import numpy as np
            from ml.inference.pipeline import pipeline
            if pipeline.model is not None:
                print("[Startup] Warming up YOLO detection model on CPU kernels...")
                dummy_img = np.full((320, 320, 3), 128, dtype=np.uint8)
                infer_ctx = torch.inference_mode() if "torch" in globals() and torch is not None else nullcontext()
                with infer_ctx:
                    pipeline.model(dummy_img, imgsz=640, verbose=False)
                print("[Startup] YOLO model warmup complete!")
        except Exception as warmup_err:
            print(f"[Startup] Warmup note: {warmup_err}")
    except Exception as e:
        print(f"Startup initialization notice: {e}")
