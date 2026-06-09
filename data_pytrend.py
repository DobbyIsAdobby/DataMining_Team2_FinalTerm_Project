import sys
import time
import random
import os
import pandas as pd
from pytrends.request import TrendReq
from datetime import datetime, timedelta

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Google Trends API 연결
pytrends = TrendReq(hl='ko', tz=540, timeout=(10, 25))

def safe_request(pytrends, kw_list, timeframe, geo='', retries=4): # 시도 횟수를 2 -> 4로 증가
    assert len(kw_list) <= 5, "한 요청당 키워드는 최대 5개까지만 가능합니다."
    
    for attempt in range(1, retries + 1):
        # 기본 딜레이도 약간 늘림
        wait = random.uniform(5, 10) * attempt
        print(f"  {wait:.1f}초 대기...(시도 {attempt}/{retries}) ({kw_list}, geo='{geo or 'WW'}')")
        time.sleep(wait)
        
        try:
            pytrends.build_payload(kw_list=kw_list, timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()
        except Exception as e:
            # 429 차단 시 : 시도 횟수에 비례하여 대기 시간을 늘림 (60초 -> 120초 -> 180초)
            if '429' in str(e):
                cool = 60 * attempt 
                print(f"  에러 : 429 IP 임시 차단됨 -> {cool}초 쿨타임 대기 후 재시도합니다.")
            else:
                cool = 15
                print(f"  에러 : {e} -> {cool}초 대기 후 재시도")
            
            time.sleep(cool)
            continue
            
        if 'isPartial' in df.columns:
            df = df.drop(columns='isPartial')
        if not df.empty:
            return df
        print(f"  재시도 : 빈 응답 - {attempt}번째 실패")
        
    print(f"  경고 : {kw_list} 빈 응답")
    return pd.DataFrame()

GROUPS = [
    {"lang": "en", "terms": ["crash", "bug", "optimization", "patch"]},
    {"lang": "ko", "terms": ["최적화", "버그", "패치", "업데이트"]},
]

TERM_META = {
    "crash":        ("en", "bug", "crash"),
    "bug":          ("en", "bug", "bug"),
    "optimization": ("en", "bug", "optimization"),
    "patch":        ("en", "bug", "patch"),
    "최적화":        ("ko", "bug", "optimization"),
    "버그":          ("ko", "bug", "bug"),
    "패치":          ("ko", "bug", "patch"),
    "업데이트":       ("ko", "bug", "update"),
    "masterpiece":  ("en", "sentiment_pos", "masterpiece"),
    "goty":         ("en", "sentiment_pos", "goty"),
    "갓겜":          ("ko", "sentiment_pos", "god_game"),
    "명작":          ("ko", "sentiment_pos", "masterpiece"),
    "trash":        ("en", "sentiment_neg", "trash"),
    "garbage":      ("en", "sentiment_neg", "garbage"),
    "refund":       ("en", "sentiment_neg", "refund"),
    "똥겜":          ("ko", "sentiment_neg", "shit_game"),
    "쓰레기":         ("ko", "sentiment_neg", "trash"),
    "환불":          ("ko", "sentiment_neg", "refund"),
}

def run_pipeline(en_brand, ko_brand):
    # 1. 기간 설정: 오늘 기준 과거 30일
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=30)
    timeframe = f"{start_dt.strftime('%Y-%m-%d')} {end_dt.strftime('%Y-%m-%d')}"
    
    print(f"\n=== {en_brand} / {ko_brand} [Recent 30 Days] : {timeframe} ===")
    
    # 2. 수집 로직
    full_cols = []
    frames = []
    
    for g in GROUPS:
        brand = en_brand if g["lang"] == "en" else ko_brand
        geo = "" if g["lang"] == "en" else "KR"
        kw = [f"{brand} {t}" for t in g["terms"]]
        full_cols += kw
        
        df = safe_request(pytrends, kw, timeframe, geo=geo)
        if not df.empty:
            frames.append(df)
            
    if frames:
        merged = pd.concat(frames, axis=1)
    else:
        merged = pd.DataFrame()

    if merged.empty:
        print("  실패 : 수집된 데이터가 없습니다")
        return None

    # 누락된 키워드 0으로 채워 구조 통일
    merged = merged.reindex(columns=full_cols, fill_value=0)
    merged = merged.fillna(0).astype(int)
    merged = merged.reset_index() # 인덱스(date)를 컬럼으로 이동
    
    # 3. 전처리 및 병합 로직
    rows = []
    # 분석 일관성을 위해 event_date를 수집 종료일(오늘)로 고정
    ev_dt = pd.Timestamp(end_dt.date())
    
    for _, r in merged.iterrows():
        d = r["date"]
        for col in merged.columns:
            if col == "date":
                continue

            term = col.split()[-1]
            lang, cat, concept = TERM_META.get(term, ("?", "?", term))

            rows.append({
                "date":             d.date(),
                "game":             ko_brand,
                "event":            "Recent 30 Days",
                "event_date":       ev_dt.date(),
                "days_from_event":  (d - ev_dt).days, 
                "lang":             lang,
                "category":         cat,
                "keyword":          concept,
                "raw_keyword":      col,
                "value":            int(r[col]),
            })

    ds = pd.DataFrame(rows)
    ds = ds.sort_values(["game", "event", "category", "keyword", "date"]).reset_index(drop=True)
    
    os.makedirs('data', exist_ok=True)
    outfile = f"data/pytrend_summary_{en_brand.replace(' ', '')}.csv"
    ds.to_csv(outfile, index=False, encoding="utf-8-sig")
    
    print(f"  저장 완료 -> {outfile} ({len(ds):,}행)")
    return ds

if __name__ == "__main__":
    en_input = input("게임 영문명 입력 (예: Crimson Desert): ").strip()
    ko_input = input("게임 국문명 입력 (예: 붉은사막): ").strip()
    
    if en_input and ko_input:
        run_pipeline(en_input, ko_input)
    else:
        print("영문명과 국문명을 모두 입력해야 합니다.")