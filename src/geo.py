# File: src/geo.py — baca GeoTIFF hasil Earth Engine ke array (hari, 200, 248)
import os, numpy as np, rioxarray
D = '/content/drive/MyDrive/karhutla_gee'
W0, N0, SCALE, H, W = 108.5, 4.5, 0.045, 200, 248

def parts(var, y):
    yearly = f'{D}/{var}_{y}.tif'
    if os.path.exists(yearly): return [yearly]
    return [f'{D}/{var}_{y}_q{i}.tif' for i in range(1, 5)]      # fire bisa berupa 4 potongan kuartal

def open_arr(p):
    da = rioxarray.open_rasterio(p)
    # pojok kiri-atas harus tepat di grid, kalau tidak data bergeser
    assert abs(float(da.x[0]) - (W0 + SCALE/2)) < 1e-3 and abs(float(da.y[0]) - (N0 - SCALE/2)) < 1e-3, p
    a = da.values
    assert H <= a.shape[-2] <= H + 2 and W <= a.shape[-1] <= W + 2, (p, a.shape)
    return a[..., :H, :W]                                         # potong kelebihan 1 kolom dari Earth Engine

def load_var(var, years=range(2020, 2025)):
    return np.concatenate([np.concatenate([open_arr(p) for p in parts(var, y)]) for y in years]).astype('float32')

def fill_nan(a, train):
    # Isi NaN dengan rata-rata sel itu di data latih (a: hari x sel). Sel kosong total -> rata-rata global.
    import warnings
    if not np.isnan(a).any(): return a
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = np.nanmean(a[train], axis=0)
    m = np.where(np.isnan(m), np.nanmean(a[train]), m)
    return np.where(np.isnan(a), m[None], a).astype('float32')
