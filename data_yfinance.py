import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def collect_finance_data(company, ticker_symbol):
    # 1. 기간 설정: 최근 30일
    # 7일 변동성(Volatility_7D) 계산 시 앞부분 NaN 발생을 막기 위해 40일 전부터 수집
    end_date = datetime.today()
    fetch_start_date = end_date - timedelta(days=40)
    target_start_date = end_date - timedelta(days=30)
    
    start_str = fetch_start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"\n[yfinance API] '{company}' 주가 데이터 수집 중... (Ticker: {ticker_symbol})")
    
    # 2. 주가 다운로드
    df = yf.download(ticker_symbol, start=start_str, end=end_str, progress=False)
    
    if df.empty:
        print(f"에러 : {company}({ticker_symbol})의 데이터를 불러오지 못했습니다. Ticker를 확인해주세요.")
        return None
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    df = df[['Close', 'Volume']].copy()
    
    # 3. 주말 제외 및 결측치 처리
    # pd.bdate_range는 주말(토, 일)을 제외한 평일(Business Day)만 생성
    # 평일 중 공휴일로 인한 휴장일은 ffill()에 의해 전일 종가로 채움
    b_days = pd.bdate_range(start=fetch_start_date, end=end_date)
    df = df.reindex(b_days)
    df = df.ffill()
    
    # 4. 식별용 컬럼 추가
    df['Company'] = company
    df['Window_Start'] = target_start_date.strftime('%Y-%m-%d')
    
    # 5. 파생 변수 생성
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Volatility_7D'] = df['Daily_Return'].rolling(window=7).std()
    
    # 인덱스 변환
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)
    
    # 6. 계산을 위해 여유분으로 가져온 과거 데이터를 잘라내고 '최근 30일'만 남김
    df = df[df['Date'] >= pd.Timestamp(target_start_date.date())].copy()
    
    # 초기 계산식(pct_change, rolling)으로 혹시 남은 NaN이 있다면 0으로 처리
    df.fillna(0, inplace=True)
    
    # 7. 저장
    os.makedirs('data', exist_ok=True)
    filename = f"data/finance_{company.replace(' ', '')}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"저장 완료 -> '{filename}' ({len(df)}일치 평일 데이터)")
    return df

if __name__ == "__main__":
    print("==================================================")
    print(" 주가 데이터 범용 수집기 (Recent 30 Days, 주말 제외)")
    print("==================================================")
    
    company_input = input("1. 게임사 이름 입력 (예: PearlAbyss): ").strip()
    ticker_input = input("2. 주식 Ticker 입력 (예: 263750.KQ): ").strip()
    
    if company_input and ticker_input:
        collect_finance_data(company_input, ticker_input)
    else:
        print("이름과 Ticker를 모두 입력해야 합니다.")