1. 프로젝트 개요
프로젝트 목적
서울시 상권 데이터를 활용하여 두 가지 사용자 서비스를 구현했습니다.

👔 사장님 모드

업종별 창업 추천 지역
상권별 매출 분석
유동인구 및 2030 비율 분석
폐업점포 비율 분석
머신러닝 기반 예상 월매출 제공

🔥 소비자 모드

서울 핫플 TOP 20
TOP 10 핫플 상세 조회
2030 방문 비율
주말 유동인구
야간 유동인구
지역별 인기 업종 분석
핵심 목표

복잡한 서울 상권 공공데이터를 일반 사용자도 쉽게 이해할 수 있는 추천 서비스로 변환


[seoul_where_we_go.pptx](https://github.com/user-attachments/files/32836516/seoul_where_we_go.pptx)

https://dataseoul.streamlit.app/
<img width="1325" height="833" alt="image" src="https://github.com/user-attachments/assets/2ff60f33-acb1-4d35-8130-2b42cf98e9ae" />
<img width="1306" height="842" alt="image" src="https://github.com/user-attachments/assets/92db1fbd-50d0-4a77-9480-16fcc6d0f0e3" />

<img width="901" height="431" alt="image" src="https://github.com/user-attachments/assets/6cb27fa7-c9c5-4a34-ba26-09262c08ac1f" />

<img width="886" height="429" alt="image" src="https://github.com/user-attachments/assets/eac72dc1-f39c-4ce9-9f02-d1cfdaa28a19" />

<img width="885" height="477" alt="image" src="https://github.com/user-attachments/assets/e082cab4-d18d-4987-b3e0-dd80b869171d" />

<img width="867" height="508" alt="image" src="https://github.com/user-attachments/assets/7503164e-52ce-4e26-acf9-3b65f2f47f57" />
<img width="845" height="512" alt="image" src="https://github.com/user-attachments/assets/d40abba1-0a94-401f-91ee-9c1738e57014" />

<img width="907" height="447" alt="image" src="https://github.com/user-attachments/assets/94d81547-1b42-4108-aece-05693e4cf5da" />
<img width="868" height="459" alt="image" src="https://github.com/user-attachments/assets/102d7a48-f09e-4d5e-ad5b-021d96978ae1" />

<img width="896" height="403" alt="image" src="https://github.com/user-attachments/assets/39254a64-e935-497e-aaee-1d854141baf0" />

<img width="876" height="447" alt="image" src="https://github.com/user-attachments/assets/fc140995-dcd6-418e-b603-a864079496e0" />


## 로컬 개발

Python 3.12에서 실행 및 검증했습니다. 저장된 모델과의 호환성을 위해
`scikit-learn` 버전을 유지하세요.

```bash
python3.12 -m venv /workspace/venvs/data_seoul
/workspace/venvs/data_seoul/bin/python -m pip install -r requirements.txt
/workspace/venvs/data_seoul/bin/python -m streamlit run src/app.py --server.headless=true --browser.gatherUsageStats=false
```

저장소 루트에서 실행하는 것을 권장합니다. 앱은 포함된 가공 데이터와 학습 모델을
사용하므로 최초 실행을 위해 전처리 또는 재학습할 필요가 없습니다.
소비자 질문 기능은 키워드 기반 데이터 조회이며 외부 AI API 키가 필요하지 않습니다.
`src/01_*.py`부터 `src/12_*.py`까지는 데이터 처리 및 학습 스크립트로,
실행하면 가공 데이터나 모델 파일을 덮어쓸 수 있습니다.

앱의 두 모드, 매출 예측, 질문 조회, 상세 선택을 검증하려면:

```bash
/workspace/venvs/data_seoul/bin/python -m unittest discover -s tests -v
```

## 코드 구성 및 검증 범위

- `src/app.py`: 사장님/소비자 화면과 공통 지도·점수 그래프·지표 카드 함수.
- `assets/styles.css`: 배경, 카드, 모드 토글 스타일. 스타일을 바꾸면 실행 중인 앱을 재시작하세요.
- `src/01_*.py`~`src/08_*.py`: 데이터 병합, 정제, 점수 계산 및 지도 좌표 생성.
- `src/09_*.py`~`src/12_*.py`: 기본 모델, 개선 모델, 시간 기준 평가, 최종 모델 학습.

번호가 붙은 스크립트는 직접 실행할 때만 처리하며, 다른 코드에서 import해도
데이터 생성이나 모델 학습을 시작하지 않습니다. 데이터·모델 경로는 각 파일의
위치를 기준으로 계산하므로 실행 디렉터리에 영향을 받지 않습니다.

개선 모델 및 시간 기준 평가의 이상치 기준은 학습 데이터에서만 계산하며,
평가 데이터는 제거하지 않습니다. 이 변경은 다음 평가 실행부터 적용됩니다.
기존 가공 CSV 및 저장된 모델을 이번 코드 정리 과정에서 재생성하지 않았습니다.

테스트는 두 모드의 주요 기능, 좌표 누락, 공백 질문, 동일값 점수 계산,
스크립트 import 시 부작용, 미래 분기 평가 데이터 유지 여부를 확인합니다.
전체 모델 재학습이나 브라우저의 실제 CSS 렌더링 검증은 포함하지 않습니다.
