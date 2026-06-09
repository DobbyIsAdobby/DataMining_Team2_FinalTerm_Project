import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

def aggregate_pytrend(trend_file_path):
    df = pd.read_csv(trend_file_path)
    df_negative = df[df['category'].isin(['bug', 'sentiment_neg'])]
    daily_trend = df_negative.groupby('date')['value'].max().reset_index()
    daily_trend.rename(columns={'value': 'Trend_Value'}, inplace=True)
    daily_trend['date'] = pd.to_datetime(daily_trend['date'])
    return daily_trend

def load_steam_data(steam_file_path):
    df = pd.read_csv(steam_file_path)
    df = df[['date', 'negative_ratio']].copy()
    df.rename(columns={'negative_ratio': 'Negative_Ratio'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df['Negative_Ratio'] = df['Negative_Ratio'] * 100 
    return df

def load_patch_data(patch_file_path):
    if not patch_file_path or not os.path.exists(patch_file_path):
        return pd.DataFrame(columns=['Date', 'Hotfix_Count'])
        
    df_patch = pd.read_csv(patch_file_path)
    df_patch['Date'] = pd.to_datetime(df_patch['Date'], format='mixed')
    
    df_patch['Is_Hotfix'] = df_patch['Patch_Title'].apply(
        lambda x: 1 if pd.isna(x) or str(x).lower() == 'no title' or 'hotfix' in str(x).lower() else 0
    )
    
    df_patch_grouped = df_patch.groupby('Date')['Is_Hotfix'].sum().reset_index()
    df_patch_grouped.rename(columns={'Is_Hotfix': 'Hotfix_Count'}, inplace=True)
    
    return df_patch_grouped

def apply_historical_pca(X_new):
    hist_file = 'FinalData_Cleaned.csv'
    if not os.path.exists(hist_file):
        raise FileNotFoundError(f"[치명적 에러] PCA 기준을 잡기 위한 '{hist_file}' 파일이 필요합니다.")
        
    df_hist = pd.read_csv(hist_file)
    X_hist = df_hist[['Negative_Ratio', 'Trend_Value']].values.astype(float)
    
    scaler_std = StandardScaler()
    X_hist_scaled = scaler_std.fit_transform(X_hist)
    
    pca = PCA(n_components=1, random_state=42)
    pca_hist = pca.fit_transform(X_hist_scaled)
    
    pca_sign = -1 if pca.components_[0][0] < 0 else 1
    pca_hist = pca_hist * pca_sign
        
    scaler_minmax = MinMaxScaler(feature_range=(0, 100))
    scaler_minmax.fit(pca_hist) 
    
    x_s = scaler_std.transform(X_new)
    x_p = pca.transform(x_s) * pca_sign
    crisis_index_scaled = scaler_minmax.transform(x_p)
    
    return x_p, crisis_index_scaled

def run_data_merger(company, app_id, en_brand, patch_filename, market_cap):
    finance_file = f"data/finance_{company.replace(' ', '')}.csv"
    steam_file = f"data/steam_sentiment_summary_{app_id}.csv"
    trend_file = f"data/pytrend_summary_{en_brand.replace(' ', '')}.csv"
    patch_file = f"data/{patch_filename}" if patch_filename else ""
    
    for f in [finance_file, steam_file, trend_file]:
        if not os.path.exists(f):
            print(f"[에러] '{f}' 파일이 존재하지 않습니다. 수집 모듈을 먼저 실행하세요.")
            return None

    df_finance = pd.read_csv(finance_file)
    df_finance['Date'] = pd.to_datetime(df_finance['Date'])
    
    df_steam = load_steam_data(steam_file)
    df_trend = aggregate_pytrend(trend_file)
    df_patch = load_patch_data(patch_file)
    
    print("\n[데이터 병합] 주식, 스팀, 트렌드, 패치 데이터를 통합합니다...")
    merged_df = pd.merge(df_finance, df_steam, left_on='Date', right_on='date', how='left').drop(columns=['date'])
    merged_df = pd.merge(merged_df, df_trend, left_on='Date', right_on='date', how='left').drop(columns=['date'])
    merged_df = pd.merge(merged_df, df_patch, on='Date', how='left')
    
    merged_df[['Negative_Ratio', 'Trend_Value']] = merged_df[['Negative_Ratio', 'Trend_Value']].ffill().bfill().fillna(0)
    merged_df['Hotfix_Count'] = merged_df['Hotfix_Count'].fillna(0)
    
    print("[PCA 연산] 기존 학습 데이터 기준으로 위기 지수를 정규화합니다...")
    features_for_pca = ['Negative_Ratio', 'Trend_Value']
    X_new = merged_df[features_for_pca].values
    
    principal_components, crisis_index_scaled = apply_historical_pca(X_new)

    merged_df['Crisis_Index'] = principal_components.flatten()
    merged_df['Crisis_Index_Scaled'] = crisis_index_scaled.flatten()
    
    # [신규 추가] 입력받은 시가총액을 데이터셋에 할당
    merged_df['Market_Cap'] = market_cap
    
    final_columns = [
        'Date', 'Company', 'Close', 'Volume', 'Window_Start', 'Daily_Return', 
        'Volatility_7D', 'Negative_Ratio', 'Trend_Value', 'Hotfix_Count', 
        'Crisis_Index', 'Crisis_Index_Scaled', 'Market_Cap'  # Market_Cap 컬럼 확정
    ]
    merged_df = merged_df[final_columns]
    
    os.makedirs('data', exist_ok=True)
    output_filename = f"data/Master_Dataset_{company.replace(' ', '')}.csv"
    merged_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"[완료] 통합 데이터셋 생성 성공 -> '{output_filename}' ({len(merged_df)}행)")
    return merged_df

if __name__ == "__main__":
    c_input = input("1. 게임사 이름 입력 (예: PearlAbyss): ").strip()
    a_input = input("2. Steam App ID 입력 (예: 3321460): ").strip()
    e_input = input("3. 게임 영문명 입력 (예: Crimson Desert): ").strip()
    p_input = input("4. 패치노트 CSV 파일명 입력 (예: patch.csv) (없으면 엔터): ").strip()
    
    # [신규 추가] 시가총액 프롬프트
    while True:
        try:
            m_input = input("5. 기업 시가총액 입력 (억 달러 단위, 예: 펄어비스=20, 넥슨=150): ").strip()
            market_cap_val = float(m_input)
            if market_cap_val >= 0:
                break
            print(" -> 0 이상의 숫자를 입력해주세요.")
        except ValueError:
            print(" -> 올바른 숫자를 입력해주세요.")

    if c_input and a_input and e_input:
        run_data_merger(c_input, a_input, e_input, p_input, market_cap_val)
    else:
        print("필수 정보를 올바르게 입력해야 합니다.")