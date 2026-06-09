import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import pytz
from typing import Set, Tuple

def get_integrated_target_dates(csv_path: str, launch_date_str: str, timezone_str: str) -> Tuple[Set[str], pd.Timestamp]:
    """
    패치/출시일 기준 핀셋 추출을 유지하되, 최근 30일 이내의 이벤트만 타겟.
    """
    local_tz = pytz.timezone(timezone_str)
    current_local_date = pd.Timestamp.now(tz=local_tz).normalize()
    
    global_min_date = current_local_date - pd.Timedelta(days=30)
    target_dates_set = set()

    # 1. 출시일 검증
    if launch_date_str:
        launch_dt = pd.to_datetime(launch_date_str, errors='coerce')
        if pd.notna(launch_dt):
            launch_dt = launch_dt.tz_localize(local_tz)
            for offset in range(4):
                target = launch_dt + pd.Timedelta(days=offset)
                # 계산된 날짜가 최근 30일 이내인 경우만 추가
                if target >= global_min_date:
                    target_dates_set.add(target.strftime('%Y-%m-%d'))

    # 2. 패치노트 CSV 검증
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        datetime_str = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
        df['Local_Datetime'] = pd.to_datetime(datetime_str, errors='coerce').dt.tz_localize('UTC').dt.tz_convert(local_tz)

        for patch_local_date in df['Local_Datetime'].dt.normalize().dropna():
            for offset in range(-3, 4):
                target = patch_local_date + pd.Timedelta(days=offset)
                # 패치 관련 날짜가 최근 30일 이내인 경우만 추가
                if target >= global_min_date:
                    target_dates_set.add(target.strftime('%Y-%m-%d'))
    else:
        print(f"  [안내] {csv_path} 파일을 찾을 수 없어 론칭 데이터만 탐색합니다.")

    return target_dates_set, global_min_date

def fetch_steam_reviews_by_lang(app_id, lang, target_dates_set, global_min_date, local_tz):
    print(f"  -> [{lang.upper()}] 리뷰 수집 중...")
    valid_reviews = []
    cursor = '*'
    num_per_page = 100
    MIN_PLAYTIME_MINUTES = 1 * 60 
    
    while True:
        url = f"https://store.steampowered.com/appreviews/{app_id}?json=1"
        params = {
            'filter': 'recent',
            'language': lang,
            'cursor': cursor,
            'num_per_page': num_per_page,
            'review_type': 'all',
            'purchase_type': 'all'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                time.sleep(5)
                continue
                
            data = response.json()
            if 'reviews' not in data or len(data['reviews']) == 0:
                break 
                
            batch_reviews = data['reviews']
            
            oldest_utc = pd.to_datetime(batch_reviews[-1]['timestamp_created'], unit='s').tz_localize('UTC')
            oldest_local = oldest_utc.tz_convert(local_tz)
            
            for review in batch_reviews:
                if review['author']['playtime_at_review'] < MIN_PLAYTIME_MINUTES:
                    continue
                    
                created_local = pd.to_datetime(review['timestamp_created'], unit='s').tz_localize('UTC').tz_convert(local_tz)
                created_date_str = created_local.strftime('%Y-%m-%d')
                
                # 타겟 세트(최근 30일 내 패치일)에 포함된 날짜만 저장
                if created_date_str in target_dates_set:
                    valid_reviews.append({
                        'date': created_date_str,
                        'voted_up': review['voted_up']
                    })
            
            cursor = data['cursor']
            time.sleep(1)
            
            if oldest_local < global_min_date:
                break
                
        except Exception as e:
            time.sleep(5)
            continue

    return valid_reviews

def run_pipeline(app_id, timezone_str, launch_date, csv_path):
    local_tz = pytz.timezone(timezone_str)
    
    target_dates_set, global_min_date = get_integrated_target_dates(csv_path, launch_date, timezone_str)
    print(f"  [Info] 최근 30일 내 패치/출시 관련 추출 대상 일자: {len(target_dates_set)}일")
    
    # 추출 대상 날짜가 없으면 API 호출 생략
    if len(target_dates_set) == 0:
        return []

    LANGUAGES = ['english', 'korean']
    all_reviews = []
    
    for lang in LANGUAGES:
        lang_reviews = fetch_steam_reviews_by_lang(app_id, lang, target_dates_set, global_min_date, local_tz)
        all_reviews.extend(lang_reviews)
        
    return all_reviews

def process_and_save_reviews(reviews_data, app_id):
    if not reviews_data:
        print(f"\n[결과] 최근 30일 내에 수집 조건(패치/출시)을 만족하는 데이터가 없습니다.")
        return None
        
    df_reviews = pd.DataFrame(reviews_data)
    
    daily_summary = df_reviews.groupby('date').agg(
        total_votes=('voted_up', 'count'),
        positive_votes=('voted_up', 'sum'),
        negative_votes=('voted_up', lambda x: (~x.astype(bool)).sum())
    ).reset_index()

    daily_summary['negative_ratio'] = daily_summary['negative_votes'] / daily_summary['total_votes']
    daily_summary['positive_ratio'] = daily_summary['positive_votes'] / daily_summary['total_votes']
    
    os.makedirs('data', exist_ok=True)
    output_path = f'data/steam_sentiment_summary_{app_id}.csv'
    
    daily_summary.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[완료] 파일 저장 성공 -> '{output_path}'")
    return daily_summary

if __name__ == "__main__":
    app_id_input = input("1. Steam App ID 입력: ").strip()
    tz_input = input("2. 타임존 입력 (예: Asia/Seoul, 미입력시 기본값): ").strip()
    launch_input = input("3. 게임 출시일 입력 (예: 2026-03-20): ").strip()
    csv_input = input("4. 패치노트 CSV 파일명 입력 (예: patch.csv) (없으면 엔터): ").strip()
    
    if not tz_input: tz_input = 'Asia/Seoul'
    
    if app_id_input.isdigit():
        raw_data = run_pipeline(app_id_input, tz_input, launch_input, csv_input)
        process_and_save_reviews(raw_data, app_id_input)
    else:
        print("App ID가 올바르지 않습니다.")