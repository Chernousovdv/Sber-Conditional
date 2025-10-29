import re
import unicodedata
import warnings
import numpy as np
import pandas as pd


# ---------- transliteration & sanitization helpers ----------
_RU_TO_EN = {
    'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'e','ж':'zh','з':'z','и':'i','й':'y',
    'к':'k','л':'l','м':'m','н':'n','о':'o','п':'p','р':'r','с':'s','т':'t','у':'u','ф':'f',
    'х':'kh','ц':'ts','ч':'ch','ш':'sh','щ':'shch','ы':'y','э':'e','ю':'yu','я':'ya'
}
# build uppercase mappings too
_RU_TO_EN.update({k.upper(): v.capitalize() for k,v in list(_RU_TO_EN.items())})

def transliterate_ru_to_en(s: str) -> str:
    """Transliterate Cyrillic chars in s to latin approximations. Leaves latin letters as-is."""
    out = []
    for ch in s:
        if ch in _RU_TO_EN:
            out.append(_RU_TO_EN[ch])
        else:
            out.append(ch)
    return "".join(out)

def sanitize_name(s: str) -> str:
    """
    Transliterate + remove special characters -> safe ascii feature name.
    Rules:
      - transliterate cyrillic to latin
      - replace any non-alnum with underscore
      - collapse multiple underscores
      - remove leading/trailing underscores
      - if starts with digit prefix with 'f_'
    """
    if s is None:
        return s
    # canonicalize unicode
    s = unicodedata.normalize("NFKD", str(s))
    # transliterate cyrillic -> latin
    s = transliterate_ru_to_en(s)
    # Replace non-alphanumeric with underscore
    s = re.sub(r'[^0-9A-Za-z]', '_', s)
    # collapse multiple underscores
    s = re.sub(r'_{2,}', '_', s)
    # trim underscores
    s = s.strip('_')
    # if empty, use fallback name
    if s == "":
        s = "feature"
    # if starts with digit, prefix
    if re.match(r'^[0-9]', s):
        s = "f_" + s
    return s

def sanitize_df_columns(df):
    """
    Return (df_copy_with_sanitized_columns, originals_to_sanitized_map, sanitized_to_original_map)
    Handles collisions by appending suffixes (_1, _2, ...).
    """
    df = df.copy()
    orig_cols = list(df.columns)
    sanitized = []
    mapping = {}
    inv_mapping = {}

    used = set()
    for col in orig_cols:
        s = sanitize_name(col)
        # ensure uniqueness
        base = s
        suffix = 1
        while s in used:
            s = f"{base}_{suffix}"
            suffix += 1
        used.add(s)
        sanitized.append(s)
        mapping[col] = s
        inv_mapping[s] = col

    df.columns = sanitized
    return df, mapping, inv_mapping

def map_back_feature_name(fname: str, inv_series_map: dict, inv_exog_map: dict) -> str:
    """
    Map heuristic feature names back:
      - EXOG_<san> => EXOG_<original exog name>
      - <series>_lag_<i> => <original series>_lag_<i>
      - otherwise try to map using both maps
    """
    if fname is None:
        return fname
    if fname.startswith("EXOG_"):
        key = fname[len("EXOG_"):]
        return "EXOG_" + inv_exog_map.get(key, key)
    m = re.match(r"^(.+)_lag_(\d+)$", fname)
    if m:
        col = m.group(1)
        lag = m.group(2)
        orig = inv_series_map.get(col, inv_exog_map.get(col, col))
        return f"{orig}_lag_{lag}"
    # fallback
    return inv_series_map.get(fname, inv_exog_map.get(fname, fname))


def extract_feature_importances_lgb(trained_regressor, feature_names: list[str]) -> np.ndarray:
    """
    Robust extraction of feature importances aligned to feature_names.
    Tries, in order:
      - sklearn-like .feature_importances_
      - lightgbm sklearn wrapper: .booster_.feature_importance(importance_type='gain') + booster_.feature_name()
      - linear .coef_ (abs)
      - pipeline unwrapping (try to unwrap last step)
    Returns ndarray of length len(feature_names).
    """
    import numpy as np
    from copy import deepcopy

    # Unwrap pipeline-like object to final estimator
    try:
        if hasattr(trained_regressor, "named_steps"):
            trained_reg = list(trained_regressor.named_steps.values())[-1]
        else:
            trained_reg = trained_regressor
    except Exception:
        trained_reg = trained_regressor

    imp = None
    # 1) sklearn-style feature_importances_
    if hasattr(trained_reg, "feature_importances_"):
        try:
            imp = np.array(trained_reg.feature_importances_, dtype=float)
        except Exception:
            imp = None

    # 2) lightgbm Booster (gain) and alignment by booster feature names
    if imp is None:
        try:
            if hasattr(trained_reg, "booster_") and trained_reg.booster_ is not None:
                # get importance array (gain is better than split for many use-cases)
                try:
                    booster = trained_reg.booster_
                    imp_arr = np.array(booster.feature_importance(importance_type='gain'), dtype=float)
                    booster_names = booster.feature_name()  # list of names in same order
                    # Now align booster_names with feature_names: create vector aligned to feature_names
                    name_to_imp = {n: float(v) for n, v in zip(booster_names, imp_arr)}
                    aligned = [name_to_imp.get(fn, 0.0) for fn in feature_names]
                    imp = np.array(aligned, dtype=float)
                except Exception:
                    imp = None
            
        except Exception:
            imp = None
    # # after getting trained_reg
    # if hasattr(trained_reg, "booster_") and trained_reg.booster_ is not None:
    #     print("booster feature names:", trained_reg.booster_.feature_name())
    # elif hasattr(trained_reg, "get_booster") and trained_reg.get_booster() is not None:
    #     try:
    #         print("booster feature names:", trained_reg.get_booster().feature_name())
    #     except Exception:
    #         pass
    # 3) linear coef_
    if imp is None and hasattr(trained_reg, "coef_"):
        try:
            coef = np.array(trained_reg.coef_, dtype=float)
            if coef.ndim == 1:
                imp = np.abs(coef)
            else:
                imp = np.mean(np.abs(coef), axis=0)
        except Exception:
            imp = None

    # fallback: zeros with same length as feature_names
    if imp is None:
        imp = np.zeros(len(feature_names), dtype=float)

    # If length mismatch, attempt to resize / pad/truncate but prefer safe alignment (pad zeros)
    if imp.shape[0] != len(feature_names):
        # If imp length smaller, pad zeros. If larger, truncate.
        new = np.zeros(len(feature_names), dtype=float)
        ncopy = min(len(new), imp.shape[0])
        new[:ncopy] = imp[:ncopy]
        imp = new

    # sanitize numeric
    imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0).astype(float)
    return imp