import os
import glob
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

def get_latest_master_file():
    file_pattern = os.path.join('data', 'Master_Dataset_*.csv')
    files = glob.glob(file_pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def run_predictor():
    print("=" * 70)
    print("Stacking 메타 모델 : 주가 급락 예측 프로그램을 시작합니다.")
    print("=" * 70)

    model_path = os.path.join('models', 'trained_Stacking.pkl')
    if not os.path.exists(model_path):
        print(f"에러 : 세팅된 경로에 모델 파일이 없습니다: '{model_path}'")
        return

    # 1. Bundle 데이터 로드 및 언패킹
    bundle = joblib.load(model_path)
    stacking_model = bundle['model']
    metrics = bundle['metrics']
    optimal_threshold = bundle['threshold']
    expected_features = bundle['features']  # 학습에 사용된 피처 목록

    target_csv = get_latest_master_file()
    if not target_csv:
        print("에러 : 'Master_Dataset_*.csv' 파일이 존재하지 않습니다.")
        return

    print(f"대상 파일 로드: '{target_csv}'")
    df = pd.read_csv(target_csv)
    
    # 2. 실시간 시계열 파생 변수(Feature Engineering) 계산

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Company', 'Date']).reset_index(drop=True)

    df['Log_Volume'] = np.log1p(df['Volume'])
    df['Log_Market_Cap'] = np.log1p(df['Market_Cap']) if 'Market_Cap' in df.columns else 0

    df['EMA_5_Return'] = df.groupby('Company')['Daily_Return'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    df['EMA_3_Neg'] = df.groupby('Company')['Negative_Ratio'].transform(lambda x: x.ewm(span=3, adjust=False).mean())
    df['Panic_Signal'] = df['Negative_Ratio'] * df['Log_Volume']
    df['Crisis_Volume_Ratio'] = df['Crisis_Index_Scaled'] * df['Log_Volume']
    df['Neg_Ratio_Pct_Chg'] = df.groupby('Company')['Negative_Ratio'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)

    # 파생 변수 계산이 끝난 후 가장 최신 데이터 추출
    latest_row = df.iloc[-1].copy()
    company_name = latest_row['Company']
    latest_date = latest_row['Date'].strftime('%Y-%m-%d')

    print(f" -> 대상 기업: {company_name} | 기준 데이터 날짜: {latest_date}")
    print("-" * 70)

    # 3. 사용자 입력 (Hotfix_Count 동적 처리)
    input_dict = latest_row.to_dict()

    # 만약 모델 학습 피처에 'Hotfix_Count'가 존재할 경우에만 질문하도록 설계
    if 'Hotfix_Count' in expected_features:
        while True:
            try:
                hotfix_input = input("어제 이 게임에 발생한 '긴급 핫픽스' 횟수는 몇 번인가요?: ").strip()
                hotfix_count = int(hotfix_input)
                if hotfix_count >= 0:
                    input_dict['Hotfix_Count'] = float(hotfix_count)
                    break
                print(" -> 0 이상의 정수를 입력해 주십시오.")
            except ValueError:
                print(" -> 올바른 숫자를 입력해 주십시오.")

    # 4. 예측 수행 (모델이 요구하는 순서대로 피처 조립)
    try:
        input_features = np.array([[input_dict[f] for f in expected_features]])
    except KeyError as e:
        print(f"\n에러 : 데이터셋에 필수 피처가 누락되었습니다: {e}")
        return

    # 확률 도출 및 모델이 스스로 찾은 최적 임계값 적용
    drop_probability = stacking_model.predict_proba(input_features)[0][1]
    is_danger = drop_probability >= optimal_threshold

    if is_danger:
        risk_status = "매우 위험 (경고)"
    else:
        risk_status = "안전"

    # ---------------------------------------------------------
    # 5. 리포트 출력
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print(" [리스크 진단 리포트 (향후 3일 예측)]")
    print(f" -> 이 모델이 폭락을 판별하는 '최적 임계값(Threshold)'은 {optimal_threshold*100:.1f}% 입니다.")
    print(f" -> 오늘 {company_name}의 주가가 향후 3일 내 급락할 예측 확률은 {drop_probability*100:.1f}% 입니다.")
    print(f"\n => 최종 진단: {risk_status}")
    print("-" * 70)
    print(" [예측 모델 신뢰도 지표 (Stacking + F0.7 최적화)]")
    print(" - 신규 게임에 대한 실전 대응력을 나타내는 일반화 성능입니다. - ")
    print(f" * 정밀도(Precision)      : {metrics.get('Precision', 0):.1%}")
    print(f" * 재현율(Recall)         : {metrics.get('Recall', 0):.1%}")
    print(f" * PR-AUC                 : {metrics.get('PR_AUC', 0):.4f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_predictor()