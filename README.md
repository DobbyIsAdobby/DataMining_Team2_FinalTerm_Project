# DataMining_Team2_FinalTerm_Project
한림대학교 데이터마이닝 기말프로젝트 팀 2조입니다.

# 주가 리스크 예측 시스템 (Stacking Model) 세팅 가이드

본 가이드는 Python 3.13 환경에서 가상환경(venv)을 구축하고, 필수 패키지를 설치하여 리스크 예측 시스템을 실행하는 전체 과정을 안내합니다.

## 1. 폴더 구조 확인
프로젝트 폴더 구조가 아래와 같이 구성되어 있는지 반드시 확인해 주세요.
*(특히 `models` 폴더 안에 `.pkl` 파일이 없으면 예측기가 작동하지 않습니다.)*

```text
[프로젝트 최상위 폴더]
 ┣ 📂 data           # 수집된 CSV 데이터가 저장되는 폴더
 ┣ 📂 models         
 ┃ ┗ 📄 trained_Stacking.pkl  # (필수) 사전 학습된 앙상블 모델
 ┣ 📄 data_yfinance.py
 ┣ 📄 data_steam.py
 ┣ 📄 data_pytrend.py
 ┣ 📄 data_merge.py
 ┣ 📄 Risk_Predictor.py
 ┗ 📄 requirements.txt
```

---

## 2. 가상환경(venv) 생성 및 활성화
프로젝트 폴더 경로에서 터미널(명령 프롬프트)을 열고 아래 명령어를 순서대로 실행합니다.

**① 가상환경 생성**
```bash
python -m venv .venv
```
*(Mac/Linux의 경우 `python3 -m venv .venv`를 입력하세요.)*

**② 가상환경 활성화**
* **Windows 사용자:**
  ```cmd
  venv\Scripts\activate
  ```
* **Mac/Linux 사용자:**
  ```bash
  source venv/bin/activate
  ```
> **성공 확인:** 터미널 입력창 맨 앞에 `(venv)`라는 글자가 표시되면 정상적으로 활성화된 것입니다.

---

## 3. 필수 패키지 설치
활성화된 가상환경 상태에서 시스템 구동에 필요한 모든 외부 라이브러리를 설치합니다. `requirements.txt` 파일이 아래 내용으로 작성되어 있는지 확인하세요.

**[requirements.txt 내용]**
```text
numpy==2.4.6
pandas==3.0.3
scikit-learn==1.8.0
imbalanced-learn==0.14.2
joblib==1.5.3
xgboost==3.2.0
lightgbm==4.6.0
catboost==1.2.10
yfinance==1.4.1
pytrends==4.9.2
requests==2.34.2
pytz==2026.2
lxml==6.1.1
matplotlib==3.10.9
seaborn==0.13.2
curl_cffi==0.15.0
beautifulsoup4==4.14.3
```

**[패키지 설치 명령어]**
의존성 충돌과 C++ 빌드 에러를 방지하기 위해 아래 명령어로 설치합니다.
```bash
python -m pip install --upgrade pip #안 하셔도 됩니다.
python -m pip install --no-cache-dir --force-reinstall -r requirements.txt
```

---

## 4. 파이프라인 실행 순서
설치가 끝났다면 아래 순서대로 스크립트를 실행하여 리스크를 예측할 수 있습니다.

### Step 0. SteamDB 패치내역 데이터 수집(반드시 최우선으로 진행해주세요.)
SteamDB사이트에 Steam계정으로 로그인 후 수집을 원하는 게임을 검색 후 해당 페이지에서 f12 -> Console에 들어가서 아래 코드를 복사 후 붙여넣어주세요.
```javascript
// 1. 패치 목록이 있는 테이블(tbody#js-builds) 확인
const tbody = document.querySelector('tbody#js-builds');

if (!tbody) {
    console.error('오류: 패치 목록을 찾을 수 없습니다. SteamDB의 "Patches" 탭인지 확인해주세요.');
} else {
    // 2. 게임 타이틀 가져오기 (CSV 파일명으로 사용)
    const gameTitleElement = document.querySelector('h1[itemprop="name"]');
    const gameTitle = gameTitleElement ? gameTitleElement.innerText.trim() : document.title.split('·')[0].trim();
    const safeFileName = gameTitle.replace(/[\\/*?"<>|]/g, '').replace(/\s+/g, '_');

    // 3. CSV 데이터 헤더 설정
    let csvContent = "Date,Day,Time,Patch_Title,Build_ID,Patch_Link\n";

    // 4. 테이블의 각 행(tr) 순회하며 데이터 추출
    const rows = tbody.querySelectorAll('tr');
    let count = 0;

    rows.forEach(row => {
        const tds = row.querySelectorAll('td');
        // 구조상 td가 7개 존재하는 유효한 행인지 확인
        if (tds.length >= 7) {
            const date = tds[0].innerText.trim();
            const day = tds[1].innerText.trim();
            const time = tds[2].innerText.trim();
            
            // 타이틀 및 링크 (a 태그가 없을 수도 있으므로 방어적 처리)
            const titleNode = tds[3].querySelector('a') || tds[3];
            let title = titleNode.innerText.trim();
            const link = titleNode.href ? titleNode.href : '';
            
            // Build ID 추출
            const buildId = tds[6].innerText.trim();

            // CSV 데이터 포맷팅: 제목에 콤마나 따옴표가 있을 경우를 대비해 이스케이프 처리
            title = title.replace(/"/g, '""');
            title = `"${title}"`;

            // 추출한 데이터를 CSV 문자열에 추가
            csvContent += `${date},${day},${time},${title},${buildId},${link}\n`;
            count++;
        }
    });

    // 5. CSV 파일 생성 및 다운로드
    // 한글 및 특수문자 깨짐 방지를 위해 UTF-8 BOM(\uFEFF) 추가
    const blob = new Blob(["\uFEFF" + csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `${safeFileName}_Patches.csv`);
    
    document.body.appendChild(link);
    link.click();
    
    // 사용이 끝난 객체 메모리 정리
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    console.log(`성공: ${gameTitle}의 패치 데이터 ${count}건이 다운로드되었습니다.`);
}
```

### Step 1. 기초 데이터 수집 (순서 무관)
아래 3개의 수집 모듈을 실행하여 주가, 검색 트렌드, 스팀 리뷰 데이터를 수집합니다.
yfinance.py : 수집을 원하는 회사의 주식 Ticker를 반드시 입력해주세요.
pytrend.py : 수집을 원하는 게임의 정확한 한/영문 이름을 입력해주세요.
steam.py : 수집을 원하는 게임의 appid와 지역, 출시일, 마지막으로 패치노트 csv 파일명(필수!)을 입력해주세요.
```bash
python data_yfinance.py
python data_pytrend.py
python data_steam.py
```

### Step 2. 데이터 병합 (Feature Engineering)
수집된 데이터를 하나로 묶고 파생 변수의 기준이 될 시가총액을 부여하여 통합 데이터셋을 생성합니다.
```bash
python data_merge.py
```
> **Tip:** 실행 중 기업명, 앱ID, 영문명, 패치 파일명, **기업 시가총액(억 달러)**을 차례대로 입력하라는 안내가 나옵니다. 정확히 입력해 주세요.

### Step 3. 리스크 예측 리포트 확인 (최종)
완성된 통합 데이터를 Stacking 모델에 통과시켜 향후 3일 내의 폭락 확률을 진단합니다.
```bash
python Risk_Predictor.py
```
> **진행 방식:** 스크립트가 실행되면 내부적으로 시계열 파생 변수를 자동 연산한 뒤, 어제 발생한 **'긴급 핫픽스 횟수'**를 질문합니다. 숫자를 입력하면 데이터 기반의 최적 임계값(Threshold)을 거친 최종 예측 리포트가 출력됩니다.

---
