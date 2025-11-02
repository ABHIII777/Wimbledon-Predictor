import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict, deque
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import math

DATA_GLOB = "data/raw/atp_matches_*.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
RECENT_N = 10
BASE_ELO = 1500.0
K_ELO = 32.0
RANDOM_STATE = 42
N_ITER = 30

def read_all_matches(glob_pattern=DATA_GLOB):
    files = sorted(glob(glob_pattern))
    if not files:
        raise FileNotFoundError("No match CSVs found with pattern: " + glob_pattern)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            year = None
            fname = os.path.basename(f)
            parts = fname.split('_')
            for p in parts:
                if p.isdigit() and len(p) == 4:
                    year = int(p)
                    break
            if year and 'tourney_date' not in df.columns and 'year' not in df.columns and 'tourney_year' not in df.columns:
                df['tourney_year'] = year
            dfs.append(df)
        except Exception as e:
            print("⚠️ Failed to read", f, e)
    return pd.concat(dfs, ignore_index=True)

def normalize_name(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = s.replace("\xa0"," ").replace(".", "").lower()
    s = " ".join(s.split())
    return s

def build_features_streaming(df):
    df.columns = [c.strip().lower() for c in df.columns]
    winner_col = next((c for c in df.columns if 'winner' in c and 'name' in c), None)
    loser_col  = next((c for c in df.columns if 'loser' in c and 'name' in c), None)
    if not winner_col or not loser_col:
        raise ValueError("Winner/Loser name columns not found")

    df['winner_norm'] = df[winner_col].apply(normalize_name)
    df['loser_norm']  = df[loser_col].apply(normalize_name)

    surface_col = next((c for c in df.columns if c == 'surface' or 'surface' in c), None)
    df['surface'] = df[surface_col].astype(str).fillna('unknown') if surface_col else 'unknown'

    if 'tourney_date' in df.columns:
        df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce')
        df = df.sort_values('tourney_date').reset_index(drop=True)
    elif 'tourney_year' in df.columns:
        df = df.sort_values('tourney_year').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    cand_stats = ['ace','df','svpt','1stin','1stwon','2ndwon','svwon','bp_faced','bp_saved','rank','rank_points','age','ht']
    available_stats = [s for s in cand_stats if f"w_{s}" in df.columns and f"l_{s}" in df.columns]

    for s in available_stats:
        for pref in ['w_','l_']:
            col = f"{pref}{s}"
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    elo_seed = {}
    elo_path = "data/raw/atp_elo.csv"
    if os.path.exists(elo_path):
        try:
            elo_df = pd.read_csv(elo_path, low_memory=False)
            namecol = next((c for c in elo_df.columns if 'player' in c.lower()), elo_df.columns[0])
            elo_df['player_norm'] = elo_df[namecol].apply(normalize_name)
            elo_cols = [c for c in elo_df.columns if 'elo' in c.lower()]
            if elo_cols:
                elo_df['elo_overall'] = elo_df[elo_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).fillna(BASE_ELO)
            else:
                elo_df['elo_overall'] = BASE_ELO
            elo_seed = dict(zip(elo_df['player_norm'], elo_df['elo_overall']))
            print("Loaded elo seed for", len(elo_seed))
        except Exception as e:
            print("Could not load elo seed:", e)

    cum_sum = defaultdict(lambda: defaultdict(float))
    cum_count = defaultdict(lambda: defaultdict(int))
    recent_results = defaultdict(lambda: deque(maxlen=RECENT_N))
    elo = defaultdict(float)
    for k,v in elo_seed.items():
        elo[k] = float(v)

    features = []
    labels = []
    meta = []

    def pre_avg(player, stat):
        s = cum_sum[player].get(stat, 0.0)
        c = cum_count[player].get(stat, 0)
        return s/c if c>0 else np.nan

    for idx, row in df.iterrows():
        w = row['winner_norm']; l = row['loser_norm']; surf = str(row.get('surface','unknown')).lower()
        w_elo_pre = elo.get(w, BASE_ELO); l_elo_pre = elo.get(l, BASE_ELO)

        feat = {}
        for s in available_stats:
            a_w = pre_avg(w, s); a_l = pre_avg(l, s)
            feat[f'diff_pref_{s}'] = (a_w - a_l) if (not pd.isna(a_w) and not pd.isna(a_l)) else np.nan

        feat['elo_diff'] = float(w_elo_pre - l_elo_pre)
        wr = np.mean(recent_results[w]) if len(recent_results[w])>0 else np.nan
        lr = np.mean(recent_results[l]) if len(recent_results[l])>0 else np.nan
        feat['recent_wr_diff'] = (wr - lr) if (not pd.isna(wr) and not pd.isna(lr)) else np.nan

        if 'winner_rank' in df.columns and 'loser_rank' in df.columns:
            r_w = pd.to_numeric(row.get('winner_rank', np.nan), errors='coerce')
            r_l = pd.to_numeric(row.get('loser_rank', np.nan), errors='coerce')
            feat['diff_pref_rank'] = (r_w - r_l) if (not pd.isna(r_w) and not pd.isna(r_l)) else np.nan

        feat['surface'] = surf
        features.append(feat); labels.append(1); meta.append((w,l,surf, idx))

        for s in available_stats:
            wcol = f"w_{s}"; lcol = f"l_{s}"
            if wcol in df.columns and not pd.isna(row.get(wcol)):
                try:
                    cum_sum[w][s] += float(row[wcol]); cum_count[w][s] += 1
                except: pass
            if lcol in df.columns and not pd.isna(row.get(lcol)):
                try:
                    cum_sum[l][s] += float(row[lcol]); cum_count[l][s] += 1
                except: pass

        recent_results[w].append(1); recent_results[l].append(0)

        expected_w = 1.0 / (1.0 + 10 ** ((l_elo_pre - w_elo_pre) / 400.0))
        expected_l = 1.0 - expected_w
        elo[w] = w_elo_pre + K_ELO * (1.0 - expected_w)
        elo[l] = l_elo_pre + K_ELO * (0.0 - expected_l)

    numeric_feature_names = [f'diff_pref_{s}' for s in available_stats] + ['elo_diff', 'recent_wr_diff']
    if any('diff_pref_rank' in f for f in features):
        if 'diff_pref_rank' not in numeric_feature_names:
            numeric_feature_names.append('diff_pref_rank')

    surfaces = sorted({feat['surface'] for feat in features if 'surface' in feat})
    surface_cols = [f"surface_{s}" for s in surfaces]

    X_rows = []
    y_rows = []
    for feat in features:
        rowv = {}
        for nf in numeric_feature_names:
            rowv[nf] = feat.get(nf, np.nan)
        surf = feat.get('surface','unknown')
        for s in surfaces:
            rowv[f"surface_{s}"] = 1.0 if surf == s else 0.0
        X_rows.append(rowv); y_rows.append(1)
        neg = {k:(-v if (k in numeric_feature_names and isinstance(v,(int,float,np.number))) else v) for k,v in rowv.items()}
        X_rows.append(neg); y_rows.append(0)

    X_df = pd.DataFrame(X_rows)
    y_arr = np.array(y_rows)

    num_cols = [c for c in X_df.columns if not c.startswith('surface_')]
    medians = X_df[num_cols].median()
    X_df[num_cols] = X_df[num_cols].fillna(medians)

    return X_df, y_arr, medians, available_stats, surfaces, cum_sum, cum_count, elo, recent_results

def train_and_tune():
    t0 = time.time()
    df = read_all_matches()
    print("Loaded matches", len(df))
    X_df, y_arr, medians, available_stats, surfaces, cum_sum, cum_count, elo, recent_results = build_features_streaming(df)
    print("Features shape:", X_df.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_arr, test_size=0.2, random_state=RANDOM_STATE, stratify=y_arr)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    xgb = XGBClassifier(objective='binary:logistic', eval_metric='logloss', tree_method='hist', use_label_encoder=False, random_state=RANDOM_STATE)

    param_dist = {
        'n_estimators': [150, 300, 500],
        'max_depth': [3,4,5,6,8],
        'learning_rate': [0.01,0.03,0.05,0.1],
        'subsample': [0.6,0.7,0.8,0.9,1.0],
        'colsample_bytree': [0.6,0.7,0.8,0.9,1.0],
        'gamma': [0, 0.1, 0.3, 1],
        'min_child_weight': [1,3,5,7]
    }

    search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=N_ITER, scoring='accuracy', cv=3, verbose=2, n_jobs=-1, random_state=RANDOM_STATE)
    print("Starting RandomizedSearchCV...")
    search.fit(X_train_scaled, y_train)
    best = search.best_estimator_
    print("Best params:", search.best_params_)

    y_pred = best.predict(X_test_scaled)
    y_prob = best.predict_proba(X_test_scaled)[:,1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cr = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("Accuracy:", acc, "AUC:", auc)

    feature_columns = X_df.columns.tolist()
    try:
        best.feature_names_in_ = np.array(feature_columns)
    except Exception:
        pass

    joblib.dump(best, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))
    player_profiles = {}
    players = set(list(cum_sum.keys()) + list(elo.keys()))
    for p in players:
        prof = {}
        for s in available_stats:
            csum = cum_sum[p].get(s, 0.0)
            ccnt = cum_count[p].get(s, 0)
            prof[s] = (csum/ccnt) if ccnt>0 else medians.get(f'diff_pref_{s}', 0.0)
        prof['elo'] = elo.get(p, BASE_ELO)
        prof['recent_wr'] = np.mean(list(recent_results[p])) if len(recent_results[p])>0 else 0.5
        player_profiles[p] = prof

    joblib.dump(player_profiles, os.path.join(MODEL_DIR, "player_stats.pkl"))

    metrics = {
        "accuracy": float(acc),
        "auc": float(auc),
        "classification_report": cr,
        "confusion_matrix": cm,
        "best_params": search.best_params_
    }
    with open(os.path.join(MODEL_DIR, "metrics_report.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model & artifacts to", MODEL_DIR, "in", time.time()-t0, "s")
    return metrics

if __name__ == "__main__":
    metrics = train_and_tune()
    print("Done. Metrics:", metrics)
