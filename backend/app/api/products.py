from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database.session import get_db
from app.models.all_models import Product, ProductionBatch
from app.schemas.all_schemas import ProductCreate, ProductResponse, BatchCreate, BatchResponse
from app.api.deps import get_current_active_user
from app.models.all_models import User

router = APIRouter()

BUILTIN_PRODUCTS = [
    ("Bottle", "BTL-001", "Line 1 - Bottling", "Translucent glass & PET container defect detection"),
    ("Cable", "CBL-002", "Line 2 - Wire Harness", "Multi-core insulated wiring cables and connectors"),
    ("Capsule", "CAP-003", "Line 3 - Packaging", "Hard gelatin and vegetable pharmaceutical capsules"),
    ("Carpet", "CPT-004", "Line 4 - Textiles", "Tufted and woven industrial carpets and fabric sheets"),
    ("Grid", "GRD-005", "Line 5 - Fabrication", "Precision woven metallic mesh grids and filters"),
    ("Hazelnut", "HZN-006", "Line 6 - Sorting", "Shelled and whole industrial food nuts"),
    ("Leather", "LTH-007", "Line 7 - Tannery", "Tanned upholstery and automotive leather sheets"),
    ("Metal Nut", "NUT-008", "Line 8 - Hardware", "Machined steel and brass threaded hexagonal nuts"),
    ("Pill", "PIL-009", "Line 9 - Pharma", "Coated medical tablets and pills"),
    ("Screw", "SCR-010", "Line 10 - Fasteners", "Countersunk and pan-head threaded machine screws"),
    ("Tile", "TIL-011", "Line 11 - Ceramics", "Polished and textured architectural ceramic tiles"),
    ("Toothbrush", "TBH-012", "Line 12 - Consumer", "Hygiene brush heads and embedded bristle bundles"),
    ("Transistor", "TRS-013", "Line 13 - Electronics", "Through-hole TO-92 and power semiconductor packages"),
    ("Wood", "WOD-014", "Line 14 - Lumber", "Milled timber and hardwood floorboards"),
    ("Zipper", "ZIP-015", "Line 15 - Apparel", "Continuous coil and tooth apparel zippers"),
    ("Other / Custom", "OTH-999", "Line 99 - General / Custom", "General non-manufacturing parts, custom items, or unsupported objects"),
]

@router.post("/seed", response_model=List[ProductResponse])
def seed_builtin_products(db: Session = Depends(get_db)):
    """Idempotently seed all 15 MVTec AD manufacturing categories + Other in built-in fashion."""
    for name, code, line, desc in BUILTIN_PRODUCTS:
        existing = db.query(Product).filter((Product.product_code == code) | (Product.name == name)).first()
        if not existing:
            p = Product(name=name, product_code=code, production_line=line, description=desc)
            db.add(p)
            db.commit()
            db.refresh(p)
            # Create default batch
            batch = ProductionBatch(batch_number=f"BATCH-{code}", product_id=p.id)
            db.add(batch)
            db.commit()
    return db.query(Product).order_by(Product.id).all()

@router.post("/", response_model=ProductResponse)
def create_product(product: ProductCreate, db: Session = Depends(get_db)):
    # Support creating products both with and without auth for seamless operational workflow
    clean_name = product.name.strip()
    code = product.product_code.strip() if product.product_code else None
    if not code:
        count = db.query(Product).count() + 1
        code = f"PRD-{clean_name[:3].upper()}-{count:03d}"

    # Check for duplicate name
    existing = db.query(Product).filter(Product.name == clean_name).first()
    if existing:
        return existing

    db_product = Product(
        name=clean_name,
        description=product.description,
        product_code=code,
        production_line=product.production_line or "Line 1 - Standard",
    )
    db.add(db_product)
    db.commit()
    db.refresh(db_product)

    # Automatically create an initial default production batch for this product
    batch_num = f"BATCH-{db_product.product_code or db_product.id}-001"
    existing_batch = db.query(ProductionBatch).filter(ProductionBatch.product_id == db_product.id).first()
    if not existing_batch:
        batch = ProductionBatch(
            batch_number=batch_num,
            product_id=db_product.id
        )
        db.add(batch)
        db.commit()

    return db_product

@router.get("/", response_model=List[ProductResponse])
def get_products(skip: int = 0, limit: int = 200, db: Session = Depends(get_db)):
    # Public endpoint: return products without requiring authentication so the
    # frontend can display the catalog in development/demo environments.
    return db.query(Product).order_by(Product.id).offset(skip).limit(limit).all()

@router.get("/{product_id}", response_model=ProductResponse)
def get_product(product_id: int, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product
