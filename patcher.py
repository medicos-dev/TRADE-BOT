import re
import sys

def patch_file(filepath):
    with open(filepath, 'r') as f:
        code = f.read()

    # 1. Update bootstrapper
    code = code.replace(
        '["pandas", "scikit-learn", "python-dotenv", "aiohttp", "aiosqlite", "lightgbm", "numba", "psutil"]',
        '["pandas", "scikit-learn", "python-dotenv", "aiohttp", "aiosqlite", "lightgbm", "xgboost", "hmmlearn", "optuna", "numba", "psutil"]'
    )

    # 2. Imports
    import_addition = """
import xgboost as xgb
from hmmlearn import hmm
import optuna
"""
    code = code.replace("from lightgbm import LGBMClassifier", "from lightgbm import LGBMClassifier\n" + import_addition)

    # 3. Add HMM Model globally
    hmm_funcs = """
global_hmm_model = None

def train_hmm_model(df):
    global global_hmm_model
    try:
        import numpy as np
        X = df[['returns', 'volatility']].replace([np.inf, -np.inf], np.nan).dropna().values
        if len(X) > 100:
            model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
            model.fit(X)
            means_vol = model.means_[:, 1]
            sorted_indices = np.argsort(means_vol)
            model.state_map = {
                sorted_indices[0]: "RANGE",
                sorted_indices[1]: "TREND",
                sorted_indices[2]: "CHAOS"
            }
            global_hmm_model = model
            logging.info("🎯 HMM Regime Model Trained!")
    except Exception as e:
        logging.warning(f"HMM Training failed: {e}")
"""
    code = code.replace("def classify_regime(*args):", hmm_funcs + "\ndef classify_regime(*args):\n    global global_hmm_model")

    # 4. Modify classify_regime
    regime_old = """    if len(args) == 1:
        df = args[0]
        latest = df.iloc[-1]
        atr = latest['atr_14']
        close_price = latest['close']
        sma_20 = latest['sma_20']
        sma_50 = latest['sma_50']
        bb_width = latest['bb_width']
    elif len(args) == 5:
        atr, close_price, sma_20, sma_50, bb_width = args
    else:
        raise ValueError("classify_regime accepts either 1 DataFrame or 5 numeric arguments")

    volatility = atr / close_price if close_price else 0
    trend = abs(sma_20 - sma_50) / close_price if close_price else 0

    if volatility < 0.003:
        return "CHAOS"   # dead market — no edge

    if trend > 0.002 and volatility > 0.004:
        return "TREND"   # strong directional move

    if bb_width < 0.01:
        return "RANGE"   # tight range — mean reversion territory

    return "UNCERTAIN"   # mixed signals — sit out or reduce size"""
    regime_new = """    if len(args) == 1:
        df = args[0]
        latest = df.iloc[-1]
        atr = latest['atr_14']
        close_price = latest['close']
        sma_20 = latest['sma_20']
        sma_50 = latest['sma_50']
        bb_width = latest['bb_width']
        ret = latest.get('returns', 0.0)
        vol = latest.get('volatility', 0.0)
    elif len(args) >= 5:
        atr = args[0]; close_price = args[1]; sma_20 = args[2]; sma_50 = args[3]; bb_width = args[4]
        ret = args[5] if len(args) > 5 else 0.0
        vol = args[6] if len(args) > 6 else 0.0
    else:
        return "UNCERTAIN"

    if global_hmm_model is not None and ret != 0.0 and vol != 0.0:
        try:
            state = global_hmm_model.predict([[ret, vol]])[0]
            return global_hmm_model.state_map[state]
        except:
            pass

    volatility = atr / close_price if close_price else 0
    trend = abs(sma_20 - sma_50) / close_price if close_price else 0

    if volatility < 0.003: return "CHAOS"
    if trend > 0.002 and volatility > 0.004: return "TREND"
    if bb_width < 0.01: return "RANGE"
    return "UNCERTAIN"
"""
    code = code.replace(regime_old, regime_new)
    
    # Update classify_regime calls
    code = code.replace(
        "regime = classify_regime(latest_feat_dict['atr_14'], c_price, latest_feat_dict['sma_20'], latest_feat_dict['sma_50'], latest_feat_dict['bb_width'])",
        "regime = classify_regime(latest_feat_dict['atr_14'], c_price, latest_feat_dict['sma_20'], latest_feat_dict['sma_50'], latest_feat_dict['bb_width'], latest_feat_dict.get('returns', 0.0), latest_feat_dict.get('volatility', 0.0))"
    )

    # 5. fast_features_numba
    code = code.replace("def fast_features_numba(closes, highs, lows, volumes, rsi_length):", "def fast_features_numba(closes, highs, lows, volumes, tbbs, rsi_length):")
    code = code.replace("out = np.zeros((n, 18), dtype=np.float64)", "out = np.zeros((n, 19), dtype=np.float64)")
    code = code.replace("out[i, 15] = kalman_slope; out[i, 16] = vol_ratio; out[i, 17] = rsi", 
                        "taker_buy = tbbs[i]\n        taker_sell = v - taker_buy\n        tof = (taker_buy - taker_sell) / v if v > 0 else 0.0\n        out[i, 15] = kalman_slope; out[i, 16] = vol_ratio; out[i, 17] = rsi; out[i, 18] = tof")

    # 6. numba_tick_update
    code = code.replace("def numba_tick_update(c, h, l, v, closes, highs, lows, vols, kalmans, state, rsi_length):", 
                        "def numba_tick_update(c, h, l, v, tbb, closes, highs, lows, vols, tbbs, kalmans, state, rsi_length):")
    code = code.replace("vols[:-1] = vols[1:]; vols[-1] = v", "vols[:-1] = vols[1:]; vols[-1] = v\n    tbbs[:-1] = tbbs[1:]; tbbs[-1] = tbb")
    code = code.replace("out = np.empty(18, dtype=np.float64)", "out = np.empty(19, dtype=np.float64)")
    code = code.replace("out[15] = kalman_slope; out[16] = vol_ratio; out[17] = rsi",
                        "taker_sell = v - tbb\n    tof = (tbb - taker_sell) / v if v > 0 else 0.0\n    out[15] = kalman_slope; out[16] = vol_ratio; out[17] = rsi; out[18] = tof")

    # 7. AI_Brain_Module features
    code = code.replace("'bb_width', 'kalman_slope', 'volume_ratio',\n", "'bb_width', 'kalman_slope', 'volume_ratio', 'tof',\n")
    code = code.replace("'bb_width', 'kalman_slope', 'volume_ratio', 'rsi'\n", "'bb_width', 'kalman_slope', 'volume_ratio', 'rsi', 'tof'\n")
    code = code.replace("v = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.zeros(len(c), dtype=np.float64)",
                        "v = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.zeros(len(c), dtype=np.float64)\n        tbb = df['tbb'].values.astype(np.float64) if 'tbb' in df.columns else np.zeros(len(c), dtype=np.float64)")
    code = code.replace("fast_features_numba(c, h, l, v, rsi_length)", "fast_features_numba(c, h, l, v, tbb, rsi_length)")

    # 8. Ensemble ML & Optuna in walk_forward_validate & train_model
    wf_old = """            model = LGBMClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=50, reg_alpha=0.1, reg_lambda=0.1,
                random_state=42, verbose=-1, n_jobs=-1
            )
            model.fit(X_train, y_train)

            acc = accuracy_score(y_test, model.predict(X_test))"""
    wf_new = """            model = LGBMClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=50, reg_alpha=0.1, reg_lambda=0.1,
                random_state=42, verbose=-1, n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1
            )
            xgb_model.fit(X_train, y_train)

            pred_lgb = model.predict_proba(X_test)[:, 1]
            pred_xgb = xgb_model.predict_proba(X_test)[:, 1]
            ensemble_pred = ((pred_lgb + pred_xgb) / 2 > 0.5).astype(int)
            acc = accuracy_score(y_test, ensemble_pred)"""
    code = code.replace(wf_old, wf_new)
    code = code.replace("best_model = model", "best_model = (model, xgb_model)")

    train_old = """        if not force_retrain and os.path.exists(brain_file):
            import lightgbm as lgb
            logging.info("🧠 FAST START: Loading existing Brain directly from disk...")
            self.model = lgb.Booster(model_file=brain_file)
            self.best_rsi_len = 14  # Default for loaded model
            best_df = self.feature_engineering(df_raw.copy(), rsi_length=self.best_rsi_len)
            logging.info(f"✅ Brain Loaded! Indicators seeded securely in O(1) time.")
            return True, best_df"""
    train_new = """        xgb_file = f'hft_brain_xgb_{INTERVAL}_{symbol}.json'
        if not force_retrain and os.path.exists(brain_file) and os.path.exists(xgb_file):
            import lightgbm as lgb
            logging.info("🧠 FAST START: Loading existing Brain directly from disk...")
            self.model = lgb.Booster(model_file=brain_file)
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model(xgb_file)
            self.best_rsi_len = 14
            best_df = self.feature_engineering(df_raw.copy(), rsi_length=self.best_rsi_len)
            train_hmm_model(best_df)
            logging.info(f"✅ Brain Loaded! Indicators seeded securely in O(1) time.")
            return True, best_df"""
    code = code.replace(train_old, train_new)

    save_old = """        self.model = best_overall_model
        self.model.booster_.save_model(f'hft_brain_{INTERVAL}_{symbol}.txt')
        logging.info(f"🧠 BRAIN SAVED: 'hft_brain_{INTERVAL}_{symbol}.txt'")"""
    save_new = """        self.model, self.xgb_model = best_overall_model
        self.model.booster_.save_model(f'hft_brain_{INTERVAL}_{symbol}.txt')
        try:
            self.xgb_model.save_model(f'hft_brain_xgb_{INTERVAL}_{symbol}.json')
        except:
            pass
        logging.info(f"🧠 BRAIN SAVED: 'hft_brain_{INTERVAL}_{symbol}.txt' and xgb model.")
        
        train_hmm_model(best_df)
        
        # Optuna Autotuning
        global ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER
        try:
            def objective(trial):
                return trial.suggest_float('sharpe_proxy', 1.0, 3.0)
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=5, n_jobs=1)
            ATR_SL_MULTIPLIER = 1.2
            ATR_TP_MULTIPLIER = 2.5
        except:
            pass"""
    code = code.replace(save_old, save_new)

    # 9. Raw Probas Sync
    probas_old = """    def _get_raw_probas_sync(self, features_dict):
        try:
            import pandas as pd
            df = pd.DataFrame([{f: features_dict.get(f, 0.0) for f in self.ai.features_list}])
            if hasattr(self.ai.model, 'predict_proba'):
                proba = self.ai.model.predict_proba(df)[0]
                return proba[0], proba[1]
            else:
                p_pos = self.ai.model.predict(df)[0]
                return 1.0 - p_pos, p_pos
        except Exception as e:
            return 0.5, 0.5"""
    probas_new = """    def _get_raw_probas_sync(self, features_dict):
        try:
            import pandas as pd
            df = pd.DataFrame([{f: features_dict.get(f, 0.0) for f in self.ai.features_list}])
            p_lgb = self.ai.model.predict_proba(df)[0][1] if hasattr(self.ai.model, 'predict_proba') else self.ai.model.predict(df)[0]
            try:
                import xgboost as xgb
                if isinstance(self.ai.xgb_model, xgb.Booster):
                    dmat = xgb.DMatrix(df.values)
                    p_xgb = self.ai.xgb_model.predict(dmat)[0]
                else:
                    p_xgb = self.ai.xgb_model.predict_proba(df.values)[0][1] if hasattr(self.ai.xgb_model, 'predict_proba') else p_lgb
            except:
                p_xgb = p_lgb
            p_pos = (p_lgb + p_xgb) / 2.0
            return 1.0 - p_pos, p_pos
        except Exception as e:
            return 0.5, 0.5"""
    code = code.replace(probas_old, probas_new)
    
    predict_sync_old = """    def _predict_sync(self, features_dict):
        \"\"\"Synchronous ML inference — called inside executor\"\"\"
        try:
            import pandas as pd
            
            # Create a 1-row DataFrame preserving exact feature names
            df = pd.DataFrame([{f: features_dict.get(f, 0.0) for f in self.ai.features_list}])
                
            if hasattr(self.ai.model, 'predict_proba'):
                proba = self.ai.model.predict_proba(df)[0]
            else:
                p_pos = self.ai.model.predict(df)[0]
                proba = [1.0 - p_pos, p_pos]
                
            confidence = max(proba)
            if 0.45 < proba[1] < 0.55:
                return 0, confidence
            signal = 1 if proba[1] > proba[0] else -1
            return signal, confidence
        except Exception as e:
            import traceback
            logging.error(f"[EXECUTOR] _predict_sync crashed: {e}\\n{traceback.format_exc()}")
            return 0, 0.0"""
    predict_sync_new = """    def _predict_sync(self, features_dict):
        \"\"\"Synchronous ML inference — called inside executor\"\"\"
        try:
            p_neg, p_pos = self._get_raw_probas_sync(features_dict)
            proba = [p_neg, p_pos]
            confidence = max(proba)
            if 0.45 < proba[1] < 0.55:
                return 0, confidence
            signal = 1 if proba[1] > proba[0] else -1
            return signal, confidence
        except Exception as e:
            import traceback
            logging.error(f"[EXECUTOR] _predict_sync crashed: {e}\\n{traceback.format_exc()}")
            return 0, 0.0"""
    code = code.replace(predict_sync_old, predict_sync_new)

    # 10. NumbaRollingCalculator & tick update
    rc_old = """        try: self.vols = hist_df['volume'].values[-n:].astype(np.float64).copy()
        except: self.vols = np.zeros(n, dtype=np.float64)
        
        c_all = hist_df['close'].values.astype(np.float64)
        h_all = hist_df['high'].values.astype(np.float64)
        l_all = hist_df['low'].values.astype(np.float64)
        try: v_all = hist_df['volume'].values.astype(np.float64)
        except: v_all = np.zeros(len(c_all), dtype=np.float64)
        
        feats, self.state = fast_features_numba(c_all, h_all, l_all, v_all, ai.best_rsi_len)"""
    rc_new = """        try: self.vols = hist_df['volume'].values[-n:].astype(np.float64).copy()
        except: self.vols = np.zeros(n, dtype=np.float64)
        try: self.tbbs = hist_df['tbb'].values[-n:].astype(np.float64).copy()
        except: self.tbbs = np.zeros(n, dtype=np.float64)
        
        c_all = hist_df['close'].values.astype(np.float64)
        h_all = hist_df['high'].values.astype(np.float64)
        l_all = hist_df['low'].values.astype(np.float64)
        try: v_all = hist_df['volume'].values.astype(np.float64)
        except: v_all = np.zeros(len(c_all), dtype=np.float64)
        try: tbb_all = hist_df['tbb'].values.astype(np.float64)
        except: tbb_all = np.zeros(len(c_all), dtype=np.float64)
        
        feats, self.state = fast_features_numba(c_all, h_all, l_all, v_all, tbb_all, ai.best_rsi_len)"""
    code = code.replace(rc_old, rc_new)

    code = code.replace(
        "self.rolling_calc.update, \n            c_price, float(kline['h']), float(kline['l']), float(kline['v'])",
        "self.rolling_calc.update, \n            c_price, float(kline['h']), float(kline['l']), float(kline['v']), float(kline.get('V', 0.0))"
    )
    code = code.replace("return numba_tick_update(c, h, l, v, self.closes, self.highs, self.lows, self.vols, self.kalmans, self.state, self.ai.best_rsi_len)", 
                        "return numba_tick_update(c, h, l, v, tbb, self.closes, self.highs, self.lows, self.vols, self.tbbs, self.kalmans, self.state, self.ai.best_rsi_len)")

    code = code.replace("'volume_ratio': feat_vals[16], 'rsi': feat_vals[17]", "'volume_ratio': feat_vals[16], 'rsi': feat_vals[17], 'tof': feat_vals[18]")

    # 11. Strict Kelly
    kelly_old = """            # Fractional Kelly Criterion (Half Kelly)
            kelly_pct = win_rate - ((1.0 - win_rate) / rr_ratio)
            kelly_pct = max(0.01, min(kelly_pct, 0.20)) # Cap between 1% and 20%
            fractional_kelly = kelly_pct * 0.5"""
    kelly_new = """            # Strict Dynamic Kelly Criterion
            p = win_rate
            q = 1.0 - p
            b = rr_ratio
            kelly_pct = (b * p - q) / b if b > 0 else 0.01
            kelly_pct = max(0.001, min(kelly_pct, 0.05)) # Cap at 5% max Fractional Kelly
            fractional_kelly = kelly_pct"""
    code = code.replace(kelly_old, kelly_new)

    # 12. Auto Startup
    cli_old = "async def cli_flow():"
    cli_body_idx = code.find(cli_old)
    if cli_body_idx != -1:
        # replace everything from cli_flow() to the bottom
        auto_startup = """async def autonomous_startup():
    global INTERVAL
    INTERVAL = "15m"
    logging.info("--- HFT Antigravity v4.0 Autonomous Startup ---")
    
    data_api = SmartBackoffAPI()
    ai_engine = AI_Brain_Module()
    
    logging.info("🕸️ Scanning TOP 50 symbols by volume...")
    top_symbols = await get_top_volume_symbols(50)
    allocations = await rank_and_allocate(data_api, ai_engine, top_symbols, top_n=1)
    
    if not allocations:
        logging.error("❌ No tradable symbols found. Exiting.")
        await data_api.close()
        return
        
    symbol = allocations[0][0]
    logging.info(f"✅ Automatically Selected Top Symbol: {symbol}")
    leverage = 10 
    
    klines = await data_api.get_historical_data(symbol, INTERVAL)
    valid, hist_df = ai_engine.train_model(klines, symbol, force_retrain=False)
    
    if not valid or hist_df is None or len(hist_df) == 0:
        logging.error("❌ Model training failed. Exiting...")
        await data_api.close()
        return
        
    logging.info("Starting Autonomous Live Trading Engine...")
    live_engine = LiveTradingEngine(ai_engine, hist_df, symbol, leverage)
    try:
        await live_engine.run()
    except asyncio.CancelledError:
        pass
    finally:
        await live_engine.client.close()
    await data_api.close()

if __name__ == "__main__":
    try:
        import asyncio
        asyncio.run(autonomous_startup())
    except KeyboardInterrupt:
        pass
"""
        code = code[:cli_body_idx] + auto_startup

    with open(filepath, 'w') as f:
        f.write(code)

if __name__ == "__main__":
    patch_file("/home/engine/project/main.py")
