import streamlit as st
import FinanceDataReader as fdr
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import concurrent.futures
import time

# 1. 페이지 설정 및 다크 테마 (UI 유지)
st.set_page_config(page_title="QUANT PRO 2026", layout="wide")
st.markdown("""
    <style>
    /* 다크 모드 강제 및 커스텀 스타일 */
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    div[data-testid="stMetricValue"] { color: #00FFAA; }
    .sunja-card { 
        background-color: #1E1E1E; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #FFD700; 
        margin-bottom: 20px; 
    }
    .strategy-box {
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. 데이터 엔진 (고속/상세 분리)
@st.cache_data(ttl=3600)
def load_inventory():
    df = fdr.StockListing('KRX')
    cap_col = 'Marcap' if 'Marcap' in df.columns else 'MarketCap'
    df['시총_억'] = (df[cap_col] / 100_000_000).fillna(0).astype(int)
    return df.rename(columns={'Code':'code', 'Name':'name', 'Market':'market'})

# [상세 분석용] 모든 지표 계산 (이평선 7개, RS, 수급)
def get_detailed_data(code):
    try:
        df = fdr.DataReader(code, start="2023-01-01") 
        if df.empty: return None
        
        # 1. 이동평균선 (5, 10, 20, 60, 120, 200, 240일선)
        for window in [5, 10, 20, 60, 120, 200, 240]:
            df[f'MA{window}'] = df['Close'].rolling(window).mean()
        
        # 2. RS 강도 (KOSPI 대비)
        kospi = fdr.DataReader('KS11', start="2023-01-01")['Close']
        # 인덱스 매칭
        common_index = df.index.intersection(kospi.index)
        df = df.loc[common_index]
        kospi = kospi.loc[common_index]
        df['RS'] = (df['Close'] / df['Close'].iloc[0]) / (kospi / kospi.iloc[0]) * 100
        
        # 3. 수급 데이터 (시뮬레이션: 외인/기관/개인 선차트용 누적 데이터)
        # *실제 서비스 시에는 증권사 API 연동 필요*
        np.random.seed(int(code) if code.isdigit() else 42) 
        df['Foreigner'] = np.random.randint(-50, 50, len(df)).cumsum()
        df['Institution'] = np.random.randint(-40, 60, len(df)).cumsum()
        df['Individual'] = np.random.randint(-30, 30, len(df)).cumsum() * -1 # 개인은 반대 성향 가정
        
        return df
    except: return None

# [스캔용] 가벼운 데이터
def fetch_scan_data(code):
    try:
        df = fdr.DataReader(code, start="2024-01-01")
        if df.empty or len(df) < 60: return None
        return df
    except: return None

# 3. 손자병법 분석 로직
def sunja_analysis(df, macro_tnx):
    analysis = []
    rs = df['RS'].iloc[-1]
    cp = df['Close'].iloc[-1]
    ma200 = df['MA200'].iloc[-1]
    
    if rs > 100: analysis.append("🚩 **도(道):** 시장보다 강한 주도주입니다. 기세가 우리 편입니다.")
    else: analysis.append("🚩 **도(道):** 시장보다 약한 종목입니다. 주력 부대 투입을 보류하십시오.")
    
    if macro_tnx < 4.0: analysis.append("☁️ **천(天):** 금리 환경이 온화하여 진격하기 좋습니다.")
    else: analysis.append("☁️ **천(天):** 고금리 역풍이 부니 방어 태세를 갖추십시오.")
    
    if cp > ma200: analysis.append("⛰️ **지(地):** 장기 이평선(고지) 위에 있어 지형적 우위에 있습니다.")
    else: analysis.append("⛰️ **지(地):** 장기 이평선 아래 늪지에 빠져 있습니다. 탈출이 급선무입니다.")
    
    return analysis

# 4. [핵심] 통합 분석 대시보드 렌더링 함수 (재사용성 극대화)
def render_dashboard(stock_name, code, market_cap):
    df = get_detailed_data(code)
    if df is None:
        st.error("데이터를 불러올 수 없습니다.")
        return

    # 매크로 데이터 (단일 호출)
    tnx = yf.download("^TNX", period="5d", progress=False)
    curr_tnx = float(tnx['Close'].iloc[-1]) if not tnx.empty else 4.0

    # 지표 계산
    hi = df['Close'].max()
    lo = df['Close'].min()
    cp = float(df['Close'].iloc[-1])
    
    # 컵 완성도 (최근 고점 이후 저점 대비 회복률)
    recent_high_idx = df['Close'].idxmax()
    handle_part = df.loc[recent_high_idx:]
    handle_low = handle_part['Close'].min()
    cup_score = ((cp - handle_low) / (hi - handle_low) * 100) if hi > handle_low else 0
    
    # 전략 타점
    pivot = hi
    target = pivot * 1.25
    stop = pivot * 0.92

    # --- UI 구성 ---
    
    # 1. 상단 메트릭
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("현재가", f"{int(cp):,}원")
    c2.metric("컵 완성도", f"{cup_score:.1f}%")
    c3.metric("RS 강도", f"{df['RS'].iloc[-1]:.1f}")
    c4.metric("시가총액", f"{market_cap:,}억")

    # 2. 손자병법
    st.markdown('<div class="sunja-card"><h4>📜 손자병법 전략 리포트</h4>', unsafe_allow_html=True)
    for line in sunja_analysis(df, curr_tnx): st.write(line)
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. 통합 차트 (3단)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2],
                       subplot_titles=("가격 & 전략 타점 & 7대 이평선", "RS 상대강도", "투자자별 수급 (선차트)"))

    # [1단] 캔들 + 이평선 + 타점
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
    
    # 7대 이평선
    ma_colors = ['#FFFFFF', '#FFFF00', '#FF9900', '#FF4B4B', '#00FF00', '#8800FF', '#0083FF']
    ma_days = [5, 10, 20, 60, 120, 200, 240]
    for ma, color in zip(ma_days, ma_colors):
        fig.add_trace(go.Scatter(x=df.index, y=df[f'MA{ma}'], name=f"MA{ma}", line=dict(width=1, color=color), opacity=0.6), row=1, col=1)
    
    # 전략 타점 (가격 표시 추가)
    fig.add_hline(y=pivot, line_color="#FF4B4B", line_width=1.5, row=1, col=1, 
                  annotation_text=f"🚩 매수: {int(pivot):,}원", annotation_position="top right", annotation_font_color="#FF4B4B")
    fig.add_hline(y=target, line_color="#3498DB", line_dash="dash", line_width=1.5, row=1, col=1,
                  annotation_text=f"💰 익절: {int(target):,}원", annotation_position="top right", annotation_font_color="#3498DB")
    fig.add_hline(y=stop, line_color="#00FFAA", line_dash="dot", line_width=1.5, row=1, col=1,
                  annotation_text=f"🛡️ 손절: {int(stop):,}원", annotation_position="bottom right", annotation_font_color="#00FFAA")

    # [2단] RS
    fig.add_trace(go.Scatter(x=df.index, y=df['RS'], name="RS", fill='tozeroy', line=dict(color='#00FFAA')), row=2, col=1)

    # [3단] 수급 (선차트: 개인 포함)
    fig.add_trace(go.Scatter(x=df.index, y=df['Foreigner'], name="외국인", mode='lines', line=dict(color='#FF4B4B')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Institution'], name="기관", mode='lines', line=dict(color='#F1C40F')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Individual'], name="개인", mode='lines', line=dict(color='#A0A0A0')), row=3, col=1)

    fig.update_layout(height=1000, template="plotly_dark", showlegend=True, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig, use_container_width=True)

# 5. 스캔 병렬 처리 워커
def process_scan(args):
    code, name, target_score, market_cap = args
    df = fetch_scan_data(code)
    if df is None: return None
    
    hi = df['Close'].max()
    lo = df['Close'].min()
    cp = df['Close'].iloc[-1]
    
    if hi > lo and hi > 0:
        score = (cp - lo) / (hi - lo) * 100
        # 피벗 근접성 (전고점 대비 -15% 이내)
        if score >= target_score and cp >= hi * 0.85:
            return {
                '종목명': name, '완성도': float(f"{score:.1f}"), '현재가': int(cp), 
                '시총(억)': market_cap, 'code': code # 코드 저장 필수
            }
    return None

# 6. 메인 앱 실행
inventory = load_inventory()

with st.sidebar:
    st.title("🏯 전략 사령부")
    menu = st.radio("메뉴 선택", ["⚔️ 개별분석", "📡 컵앤핸들 스캔", "🌐 정책 & 매크로"])
    min_cap = st.number_input("최소 시총(억)", value=2000, step=500)
    filtered_df = inventory[inventory['시총_억'] >= min_cap]
    st.caption(f"분석 대상: {len(filtered_df)}개")

# --- 메뉴 1: 개별분석 ---
if menu == "⚔️ 개별분석":
    sel_name = st.selectbox("종목 검색", filtered_df['name'].tolist())
    row = filtered_df[filtered_df['name'] == sel_name].iloc[0]
    render_dashboard(row['name'], row['code'], row['시총_억'])

# --- 메뉴 2: 컵앤핸들 스캔 (통합) ---
elif menu == "📡 컵앤핸들 스캔":
    st.header("⚡ 고속 병렬 스캔 & 통합 분석")
    target_score = st.slider("패턴 완성도 기준 (%)", 60, 95, 80)
    
    # 스캔 세션 상태 관리
    if 'scan_results' not in st.session_state: st.session_state.scan_results = None

    if st.button("🚀 스캔 시작 (Parallel)"):
        targets = filtered_df.sort_values('시총_억', ascending=False)
        scan_args = [(r['code'], r['name'], target_score, r['시총_억']) for _, r in targets.iterrows()]
        
        results = []
        bar = st.progress(0)
        status = st.empty()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(process_scan, arg): arg for arg in scan_args}
            total = len(futures)
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                res = future.result()
                if res: results.append(res)
                if i % 10 == 0: bar.progress((i+1)/total)
                
        bar.progress(1.0)
        status.success(f"완료! {len(results)}개 종목 발견")
        
        if results:
            st.session_state.scan_results = pd.DataFrame(results).sort_values('완성도', ascending=False)
    
    # 결과가 있을 경우 화면 표시
    if st.session_state.scan_results is not None:
        result_df = st.session_state.scan_results
        st.dataframe(result_df, use_container_width=True)
        
        st.divider()
        st.subheader("🔍 스캔 종목 정밀 진단")
        
        # [핵심] 스캔 결과 내에서 선택 -> 개별분석 뷰 호출
        scan_list = result_df['종목명'].tolist()
        sel_scan = st.selectbox("종목을 선택하면 상세 분석이 아래에 표시됩니다.", ["선택..."] + scan_list)
        
        if sel_scan != "선택...":
            # 선택된 종목 정보 찾기
            sel_row = result_df[result_df['종목명'] == sel_scan].iloc[0]
            # 개별 분석 함수 재사용 (코드 중복 없이 완벽히 동일한 UI 제공)
            render_dashboard(sel_row['종목명'], sel_row['code'], sel_row['시총(억)'])

# --- 메뉴 3: 매크로 ---
elif menu == "🌐 정책 & 매크로":
    st.header("🌐 글로벌 매크로 지표")
    col1, col2 = st.columns(2)
    tnx = yf.download("^TNX", period="1y", progress=False)['Close']
    sox = yf.download("^SOX", period="1y", progress=False)['Close']
    
    with col1:
        st.subheader("미국 10년물 국채")
        st.line_chart(tnx)
    with col2:
        st.subheader("필라델피아 반도체")
        st.line_chart(sox)