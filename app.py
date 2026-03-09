import streamlit as st
import pandas as pd
import joblib

# 1. 페이지 설정
st.set_page_config(page_title="서울 아파트 가격 예측기", page_icon="🏠")

# 2. 모델 및 인코더 불러오기 (캐싱 처리로 속도 최적화)
@st.cache_resource
def load_ml_components():
    model = joblib.load('rf_model.pkl')
    encoder = joblib.load('district_encoder.pkl')
    return model, encoder

try:
    model, encoder = load_ml_components()
except FileNotFoundError:
    st.error("⚠️ 모델 파일(rf_model.pkl) 또는 인코더 파일(district_encoder.pkl)을 찾을 수 없습니다.")
    st.stop()

# 3. 사용자 인터페이스(UI) 구성
st.title("🔮 서울 아파트 실거래가 예측")
st.markdown("학습된 **Random Forest** 모델을 사용하여 아파트 예상 가격을 산출합니다.")

with st.container():
    st.subheader("📍 아파트 정보 입력")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 인코더에서 학습된 자치구 목록 가져오기
        clean_districts = [str(x) for x in encoder.classes_ if pd.notnull(x) and str(x) != 'nan']
        district = st.selectbox("자치구 선택", options=sorted(clean_districts))
        area = st.number_input("전용면적 (㎡)", min_value=10.0, max_value=300.0, value=84.0, step=0.1)
        year = st.number_input("건축년도", min_value=1960, max_value=2026, value=2015)

    with col2:
        floor = st.number_input("층수", min_value=-1, max_value=70, value=10)
        month = st.slider("거래 희망 월", 1, 12, 6)
        age = 2026 - year # 건물 나이 계산

# 4. 예측 실행
if st.button("💰 예상 가격 확인하기", use_container_width=True):
    # 1. 인코딩 및 기본 계산
    dist_code = encoder.transform([district])[0]
    age = 2026 - year
    
    # 2. 누락된 파생 변수들을 입력값 기반으로 계산
    high_floor = 1 if floor >= 15 else 0  # 고층여부
    new_build = 1 if age <= 10 else 0     # 신축여부
    quarter = ((month - 1) // 3 + 1)      # 분기
    
    # 면적범주 계산 (기존 실습 기준: [0,40,60,85,105,135,300])
    bins = [0, 40, 60, 85, 105, 135, 300]
    area_cat = 0.0 # 기본값
    for i in range(len(bins)-1):
        if bins[i] < area <= bins[i+1]:
            area_cat = float(i)
            break

    # 3. 모델이 학습할 때 사용한 '정확한 피처 순서'대로 데이터프레임 생성
    # 에러 메시지에 나온 순서와 모델 학습 시 순서를 반드시 맞춰야 합니다.
    feature_names = [
        '전용면적', '건축년도', '층', '월', '건물나이', 
        '면적범주', '고층여부', '신축여부', '분기', '자치구코드'
    ]
    
    input_values = [[
        area, year, floor, month, age, 
        area_cat, high_floor, new_build, quarter, dist_code
    ]]
    
    input_df = pd.DataFrame(input_values, columns=feature_names)
    
    # 4. 예측 수행
    prediction = model.predict(input_df)[0]
    
    # 결과 출력 (이후 코드는 동일)
    st.divider()
    st.balloons()
    if month in [12, 1, 2]:
        st.snow()
    st.metric(label="예상 거래가 (억원)", value=f"{prediction/10000:.2f} 억원")

# 5. 하단 설명
st.caption("본 예측값은 학습된 데이터 내의 패턴에 기반하며, 실제 시장 상황과는 차이가 있을 수 있습니다.")