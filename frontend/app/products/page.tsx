"use client";

import { useEffect, useState, useCallback, useMemo } from 'react';
import DashboardLayout from '@/components/layout/DashboardLayout';
import { productsService, CreateProductInput } from '@/services/products';
import { formatApiError } from '@/services/api';
import { Product } from '@/types';
import { 
  Plus, RotateCw, AlertCircle, CheckCircle2, Package, X, Loader2, 
  Search, Sparkles, Database, Filter, Tag, ArrowRight
} from 'lucide-react';

interface ProductFormData {
  name: string;
  product_code: string;
  production_line: string;
  description: string;
}

const initialFormData: ProductFormData = {
  name: '',
  product_code: '',
  production_line: '',
  description: '',
};

// 15 Built-in MVTec Manufacturing Categories + Other
const BUILTIN_TEMPLATES = [
  { name: 'Bottle', code: 'BTL-001', line: 'Line 1 - Bottling', desc: 'Translucent glass & PET container defect detection' },
  { name: 'Cable', code: 'CBL-002', line: 'Line 2 - Wire Harness', desc: 'Multi-core insulated wiring cables and connectors' },
  { name: 'Capsule', code: 'CAP-003', line: 'Line 3 - Packaging', desc: 'Hard gelatin and vegetable pharmaceutical capsules' },
  { name: 'Carpet', code: 'CPT-004', line: 'Line 4 - Textiles', desc: 'Tufted and woven industrial carpets and fabric sheets' },
  { name: 'Grid', code: 'GRD-005', line: 'Line 5 - Fabrication', desc: 'Precision woven metallic mesh grids and filters' },
  { name: 'Hazelnut', code: 'HZN-006', line: 'Line 6 - Sorting', desc: 'Shelled and whole industrial food nuts' },
  { name: 'Leather', code: 'LTH-007', line: 'Line 7 - Tannery', desc: 'Tanned upholstery and automotive leather sheets' },
  { name: 'Metal Nut', code: 'NUT-008', line: 'Line 8 - Hardware', desc: 'Machined steel and brass threaded hexagonal nuts' },
  { name: 'Pill', code: 'PIL-009', line: 'Line 9 - Pharma', desc: 'Coated medical tablets and pills' },
  { name: 'Screw', code: 'SCR-010', line: 'Line 10 - Fasteners', desc: 'Countersunk and pan-head threaded machine screws' },
  { name: 'Tile', code: 'TIL-011', line: 'Line 11 - Ceramics', desc: 'Polished and textured architectural ceramic tiles' },
  { name: 'Toothbrush', code: 'TBH-012', line: 'Line 12 - Consumer', desc: 'Hygiene brush heads and embedded bristle bundles' },
  { name: 'Transistor', code: 'TRS-013', line: 'Line 13 - Electronics', desc: 'Through-hole TO-92 and power semiconductor packages' },
  { name: 'Wood', code: 'WOD-014', line: 'Line 14 - Lumber', desc: 'Milled timber and hardwood floorboards' },
  { name: 'Zipper', code: 'ZIP-015', line: 'Line 15 - Apparel', desc: 'Continuous coil and tooth apparel zippers' },
  { name: 'Other / Custom Object', code: 'OTH-999', line: 'Line 99 - General / Custom', desc: 'Custom manufacturing parts, unsupported items, or general objects' },
];

const CATEGORY_CHIPS = [
  'All', 'Bottle', 'Cable', 'Capsule', 'Carpet', 'Grid', 'Hazelnut', 
  'Leather', 'Metal Nut', 'Pill', 'Screw', 'Tile', 'Toothbrush', 
  'Transistor', 'Wood', 'Zipper', 'Other'
];

export default function ProductsPage() {
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isSeeding, setIsSeeding] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState<string | null>(null);
  const [formData, setFormData] = useState<ProductFormData>(initialFormData);
  
  // Search and Filter State
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');

  const fetchProducts = useCallback(async () => {
    try {
      setLoading(true);
      setFetchError(null);
      const data = await productsService.getAll(0, 200);
      setProducts(Array.isArray(data) ? data : []);
    } catch (error: any) {
      console.error('[ProductsPage] Failed to fetch products:', error);
      const formatted = formatApiError(error, 'Failed to load products from server.');
      setFetchError(formatted);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchProducts();
  }, [fetchProducts]);

  const handleSeedBuiltin = async () => {
    try {
      setIsSeeding(true);
      setSaveError(null);
      const seeded = await productsService.seedBuiltin();
      if (Array.isArray(seeded) && seeded.length > 0) {
        setProducts(seeded);
        setSaveSuccess(`Built-in catalog seeded successfully! Total products: ${seeded.length}`);
      } else {
        await fetchProducts();
        setSaveSuccess('Built-in catalog verified and refreshed in database.');
      }
      setTimeout(() => setSaveSuccess(null), 4000);
    } catch (err: any) {
      console.error('[ProductsPage] Seeding error:', err);
      setSaveError(formatApiError(err, 'Failed to seed built-in products.'));
    } finally {
      setIsSeeding(false);
    }
  };

  const handleOpenModal = () => {
    setFormData(initialFormData);
    setSaveError(null);
    setSaveSuccess(null);
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    if (isSaving) return;
    setIsModalOpen(false);
    setSaveError(null);
    setFormData(initialFormData);
  };

  const handleSelectTemplate = (templateName: string) => {
    const tmpl = BUILTIN_TEMPLATES.find((t) => t.name === templateName);
    if (tmpl) {
      setFormData({
        name: tmpl.name,
        product_code: tmpl.code,
        production_line: tmpl.line,
        description: tmpl.desc,
      });
      setSaveError(null);
    }
  };

  const handleSave = async (e?: React.FormEvent | React.MouseEvent) => {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }

    if (isSaving) return;

    if (!formData.name || !formData.name.trim()) {
      setSaveError('Product Name is required.');
      return;
    }

    try {
      setIsSaving(true);
      setSaveError(null);

      const payload: CreateProductInput = {
        name: formData.name.trim(),
        product_code: formData.product_code?.trim() || undefined,
        production_line: formData.production_line?.trim() || undefined,
        description: formData.description?.trim() || undefined,
      };

      const createdProduct = await productsService.create(payload);

      if (createdProduct && createdProduct.id) {
        setProducts((prev) => [
          createdProduct,
          ...prev.filter((p) => p.id !== createdProduct.id)
        ]);
      }

      setSaveSuccess(`Product "${createdProduct.name}" (ID: #${createdProduct.id}) created successfully!`);
      setTimeout(() => setSaveSuccess(null), 4000);

      setIsModalOpen(false);
      setFormData(initialFormData);

      await fetchProducts();
    } catch (error: any) {
      console.error('[ProductsPage] Failed to create product:', error);
      const formatted = formatApiError(error, 'Failed to save product.');
      setSaveError(formatted);
    } finally {
      setIsSaving(false);
    }
  };

  // Filtered Products
  const filteredProducts = useMemo(() => {
    return products.filter((p) => {
      const q = searchQuery.toLowerCase().trim();
      const matchesSearch = 
        !q ||
        p.id.toString().includes(q) ||
        p.name.toLowerCase().includes(q) ||
        (p.product_code && p.product_code.toLowerCase().includes(q)) ||
        (p.production_line && p.production_line.toLowerCase().includes(q));

      if (!matchesSearch) return false;

      if (selectedCategory === 'All') return true;

      const normName = p.name.toLowerCase();
      const normCat = selectedCategory.toLowerCase().replace(' ', '_');

      if (selectedCategory === 'Other') {
        return normName.includes('other') || normName.includes('unknown') || normName.includes('custom');
      }

      return normName.includes(normCat) || normName.includes(selectedCategory.toLowerCase());
    });
  }, [products, searchQuery, selectedCategory]);

  return (
    <DashboardLayout>
      {/* Header section */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
        <div>
          <h1 className="text-2xl font-bold text-slate-800 flex items-center gap-2.5">
            <Package className="text-blue-600" size={28} />
            <span>Product Catalog & IDs</span>
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Built-in 15 MVTec AD manufacturing datasets, custom catalog items, and production line mapping
          </p>
        </div>

        <div className="flex items-center flex-wrap gap-2.5">
          <button
            type="button"
            onClick={handleSeedBuiltin}
            disabled={isSeeding || loading}
            className="px-3.5 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl transition-all flex items-center gap-2 text-xs font-bold shadow-xs cursor-pointer disabled:opacity-50"
            title="Automatically ensure all 15 MVTec categories + Other exist in the database in less time"
          >
            {isSeeding ? (
              <Loader2 size={16} className="animate-spin text-blue-600" />
            ) : (
              <Sparkles size={16} className="text-blue-600" />
            )}
            <span>Seed Built-in (15 MVTec + Other)</span>
          </button>

          <button
            type="button"
            onClick={() => fetchProducts()}
            disabled={loading}
            className="p-2.5 border border-slate-200 hover:bg-slate-50 text-slate-600 rounded-xl transition-colors flex items-center gap-1.5 text-xs font-semibold cursor-pointer"
            title="Refresh product list"
          >
            <RotateCw size={15} className={loading ? 'animate-spin text-blue-600' : ''} />
            <span>Refresh</span>
          </button>

          <button
            type="button"
            id="add-product-btn"
            onClick={handleOpenModal}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2.5 rounded-xl font-bold flex items-center gap-2 shadow-sm transition-all text-xs cursor-pointer"
          >
            <Plus size={18} />
            <span>Add Product</span>
          </button>
        </div>
      </div>

      {saveSuccess && (
        <div className="mb-4 p-4 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-xl flex items-center gap-3 shadow-sm animate-fadeIn">
          <CheckCircle2 size={20} className="text-emerald-600 shrink-0" />
          <span className="text-sm font-medium">{saveSuccess}</span>
        </div>
      )}

      {fetchError && (
        <div className="mb-4 p-4 bg-rose-50 border border-rose-200 text-rose-800 rounded-xl flex items-center justify-between shadow-sm">
          <div className="flex items-center gap-3">
            <AlertCircle size={20} className="text-rose-600 shrink-0" />
            <span className="text-sm font-medium">{fetchError}</span>
          </div>
          <button
            type="button"
            onClick={() => fetchProducts()}
            className="text-xs font-semibold text-rose-700 underline hover:text-rose-900 cursor-pointer"
          >
            Retry
          </button>
        </div>
      )}

      {/* Filter and search bar */}
      <div className="bg-white p-4 rounded-xl shadow-xs border border-slate-200 mb-6 space-y-3">
        <div className="flex flex-col sm:flex-row gap-3 items-center justify-between">
          <div className="relative w-full sm:w-80">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
              <Search size={16} />
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by ID, name, code, line..."
              className="w-full pl-9 pr-4 py-2 text-xs border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-600 text-slate-800 placeholder:text-slate-400"
            />
            {searchQuery && (
              <button 
                onClick={() => setSearchQuery('')}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-600 text-xs"
              >
                Clear
              </button>
            )}
          </div>

          <div className="text-xs text-slate-500 font-medium">
            Showing <span className="font-bold text-slate-800">{filteredProducts.length}</span> of <span className="font-bold text-slate-800">{products.length}</span> products
          </div>
        </div>

        {/* Category Pills */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 text-xs">
          <div className="flex items-center gap-1 text-slate-400 shrink-0 mr-1">
            <Filter size={13} />
            <span className="font-semibold uppercase text-[10px] tracking-wider">Filter:</span>
          </div>
          {CATEGORY_CHIPS.map((cat) => {
            const active = selectedCategory === cat;
            return (
              <button
                key={cat}
                type="button"
                onClick={() => setSelectedCategory(cat)}
                className={`px-2.5 py-1 rounded-lg font-medium text-xs whitespace-nowrap transition-all cursor-pointer ${
                  active
                    ? 'bg-blue-600 text-white font-bold shadow-xs'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                {cat}
              </button>
            );
          })}
        </div>
      </div>

      {/* Products Table */}
      <div className="bg-white rounded-xl shadow-xs border border-slate-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-slate-500 font-semibold text-xs uppercase tracking-wider">
                <th className="p-4 w-24">Product ID</th>
                <th className="p-4 w-36">Product Code</th>
                <th className="p-4">Product Name</th>
                <th className="p-4">Category / Type</th>
                <th className="p-4">Production Line</th>
                <th className="p-4">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-sm">
              {loading ? (
                <tr>
                  <td colSpan={6} className="p-12 text-center text-slate-500">
                    <div className="flex flex-col items-center justify-center gap-2">
                      <Loader2 className="animate-spin text-blue-600" size={24} />
                      <span className="text-slate-600 font-medium text-xs">Loading database product catalog...</span>
                    </div>
                  </td>
                </tr>
              ) : filteredProducts.length === 0 ? (
                <tr>
                  <td colSpan={6} className="p-12 text-center text-slate-500">
                    <div className="flex flex-col items-center justify-center gap-2">
                      <Package className="text-slate-300" size={36} />
                      <p className="font-semibold text-slate-700">No products matching filter</p>
                      <p className="text-xs text-slate-400">
                        {searchQuery || selectedCategory !== 'All' 
                          ? 'Try changing search query or category filter.' 
                          : 'Click "Seed Built-in (15 MVTec + Other)" to populate standard products in 1 click.'}
                      </p>
                    </div>
                  </td>
                </tr>
              ) : (
                filteredProducts.map((product) => {
                  const lowerName = product.name.toLowerCase();
                  const isOther = lowerName.includes('other') || lowerName.includes('custom') || lowerName.includes('unknown');
                  const isMvtec = [
                    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
                    'leather', 'metal_nut', 'metal nut', 'pill', 'screw', 'tile',
                    'toothbrush', 'transistor', 'wood', 'zipper'
                  ].some(k => lowerName.includes(k));

                  return (
                    <tr key={product.id} className="hover:bg-blue-50/30 transition-colors group">
                      <td className="p-4">
                        <span className="inline-flex items-center px-2.5 py-1 rounded-md bg-blue-100 text-blue-800 font-mono text-xs font-bold shadow-2xs">
                          ID: #{product.id}
                        </span>
                      </td>
                      <td className="p-4">
                        {product.product_code ? (
                          <span className="inline-block px-2 py-0.5 bg-slate-100 text-slate-800 font-mono text-xs font-semibold rounded border border-slate-200">
                            {product.product_code}
                          </span>
                        ) : (
                          <span className="text-slate-400 text-xs">-</span>
                        )}
                      </td>
                      <td className="p-4 font-bold text-slate-900 flex items-center gap-2">
                        <span>{product.name}</span>
                      </td>
                      <td className="p-4">
                        {isOther ? (
                          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-amber-100 text-amber-800">
                            <Tag size={11} />
                            Other / Custom
                          </span>
                        ) : isMvtec ? (
                          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-emerald-100 text-emerald-800">
                            <Tag size={11} />
                            MVTec AD Core
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-slate-100 text-slate-700">
                            General
                          </span>
                        )}
                      </td>
                      <td className="p-4 text-slate-600 text-xs">
                        {product.production_line ? (
                          <span className="inline-block px-2.5 py-1 bg-slate-100 text-slate-700 text-xs font-medium rounded-md">
                            {product.production_line}
                          </span>
                        ) : (
                          <span className="text-slate-400">-</span>
                        )}
                      </td>
                      <td className="p-4 text-slate-500 text-xs max-w-xs truncate" title={product.description || ''}>
                        {product.description || <span className="text-slate-400">-</span>}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Add Product Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center z-50 p-4 animate-fadeIn">
          <div className="bg-white rounded-2xl shadow-xl w-full max-w-md border border-slate-100 overflow-hidden transform transition-all">
            <div className="flex items-center justify-between p-6 border-b border-slate-100 bg-slate-50/50">
              <div className="flex items-center gap-2.5">
                <div className="w-9 h-9 rounded-lg bg-blue-50 text-blue-600 flex items-center justify-center font-bold">
                  <Plus size={20} />
                </div>
                <div>
                  <h2 className="text-lg font-bold text-slate-900">Add New Product</h2>
                  <p className="text-xs text-slate-500">Enter details or select a built-in template</p>
                </div>
              </div>
              <button
                type="button"
                onClick={handleCloseModal}
                disabled={isSaving}
                className="text-slate-400 hover:text-slate-600 p-1.5 rounded-lg hover:bg-slate-100 transition-colors disabled:opacity-50 cursor-pointer"
              >
                <X size={18} />
              </button>
            </div>

            <form onSubmit={handleSave} className="p-6 space-y-4">
              {/* Quick Template Picker for fast creation in less time */}
              <div className="p-3 bg-blue-50/60 rounded-xl border border-blue-100">
                <label className="block text-[11px] font-bold text-blue-900 uppercase tracking-wider mb-1 flex items-center gap-1.5">
                  <Sparkles size={13} className="text-blue-600" />
                  <span>Quick Built-in Template (Auto-fills in 1 Click)</span>
                </label>
                <select
                  onChange={(e) => {
                    if (e.target.value) handleSelectTemplate(e.target.value);
                  }}
                  defaultValue=""
                  className="w-full px-3 py-2 text-xs border border-blue-200 rounded-lg bg-white font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                >
                  <option value="">-- Select Template (Bottle, Cable, Capsule, Other...) --</option>
                  {BUILTIN_TEMPLATES.map((t) => (
                    <option key={t.name} value={t.name}>
                      {t.name} ({t.code})
                    </option>
                  ))}
                </select>
              </div>

              {saveError && (
                <div className="p-3.5 bg-rose-50 border border-rose-200 text-rose-800 rounded-xl text-xs flex items-start gap-2.5">
                  <AlertCircle size={16} className="text-rose-600 shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="font-semibold">Unable to Save</p>
                    <p className="mt-0.5">{saveError}</p>
                  </div>
                </div>
              )}

              <div>
                <label htmlFor="product-name" className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
                  Product Name <span className="text-rose-500">*</span>
                </label>
                <input
                  id="product-name"
                  type="text"
                  value={formData.name}
                  onChange={(e) => {
                    setFormData((prev) => ({ ...prev, name: e.target.value }));
                    if (saveError) setSaveError(null);
                  }}
                  disabled={isSaving}
                  required
                  className="w-full px-3.5 py-2.5 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-600 transition-all disabled:bg-slate-50 text-slate-800"
                  placeholder="e.g. Bottle, Cable, or Other"
                />
              </div>

              <div>
                <label htmlFor="product-code" className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
                  Product Code / SKU
                </label>
                <input
                  id="product-code"
                  type="text"
                  value={formData.product_code}
                  onChange={(e) => setFormData((prev) => ({ ...prev, product_code: e.target.value }))}
                  disabled={isSaving}
                  className="w-full px-3.5 py-2.5 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-600 transition-all disabled:bg-slate-50 font-mono text-xs text-slate-800"
                  placeholder="e.g. BTL-001 or OTH-999"
                />
              </div>

              <div>
                <label htmlFor="production-line" className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
                  Production Line
                </label>
                <input
                  id="production-line"
                  type="text"
                  value={formData.production_line}
                  onChange={(e) => setFormData((prev) => ({ ...prev, production_line: e.target.value }))}
                  disabled={isSaving}
                  className="w-full px-3.5 py-2.5 text-sm border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-600 transition-all disabled:bg-slate-50 text-slate-800"
                  placeholder="e.g. Line 1 - Bottling"
                />
              </div>

              <div>
                <label htmlFor="product-description" className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-1.5">
                  Description
                </label>
                <textarea
                  id="product-description"
                  value={formData.description}
                  onChange={(e) => setFormData((prev) => ({ ...prev, description: e.target.value }))}
                  disabled={isSaving}
                  rows={2}
                  className="w-full px-3.5 py-2 text-xs border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-600 transition-all disabled:bg-slate-50 resize-none text-slate-800"
                  placeholder="Optional product description or defect specifications..."
                />
              </div>

              <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-100">
                <button
                  type="button"
                  id="cancel-product-btn"
                  onClick={handleCloseModal}
                  disabled={isSaving}
                  className="px-4 py-2 text-xs font-semibold text-slate-600 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition-colors disabled:opacity-50 cursor-pointer"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  id="save-product-btn"
                  disabled={isSaving}
                  className="px-5 py-2 text-xs font-bold bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors shadow-sm disabled:opacity-50 flex items-center gap-2 cursor-pointer"
                >
                  {isSaving ? (
                    <>
                      <Loader2 size={15} className="animate-spin" />
                      <span>Saving Product...</span>
                    </>
                  ) : (
                    'Save Product'
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </DashboardLayout>
  );
}
