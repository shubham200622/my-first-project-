import streamlit as st
import pandas as pd
import plotly.express as px
import time

from hybrid_data_engine import get_hybrid_live_data
from sentiment_ai import analyze_sentiment_list
from emotion_ai import detect_emotion_list
from ai_decision_engine import ai_decision

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Live Social Media Sentiment Analysis Dashboard using AI & NLP",
    page_icon="🔥",
    layout="wide"
)

# ================= HEADER =================
st.markdown(
    "<h1 style='text-align:center;'>🔥 Live Social Media Sentiment Analysis Dashboard using AI & NLP 🔥</h1>",
    unsafe_allow_html=True
)

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Control Panel")
keyword = st.sidebar.text_input("🔍 Keyword / Topic", "AI")
num_comments = st.sidebar.slider("💬 Number of Live Data", 5, 30, 10)
auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh")
run = st.sidebar.button("🚀 Start Analysis")

st.sidebar.markdown("### 🔗 Data Sources")
st.sidebar.write("• Live Web Data")
st.sidebar.write("• Live Reddit Data")
st.sidebar.write("• Hybrid Intelligence Engine")

# ================= MAIN =================
if run:
    st.success("System Started Successfully 🚀")

    # ===== HYBRID LIVE DATA =====
    comments = get_hybrid_live_data(keyword, num_comments)

    sentiments, confidences = analyze_sentiment_list(comments)
    emotions = detect_emotion_list(comments)

    df = pd.DataFrame({
        "Text": comments,
        "Sentiment": sentiments,
        "Confidence": confidences,
        "Emotion": emotions
    })

    # ===== METRICS =====
    pos = sentiments.count("Positive")
    neg = sentiments.count("Negative")
    neu = sentiments.count("Neutral")

    c1, c2, c3 = st.columns(3)
    c1.metric("🙂 Positive", pos)
    c2.metric("😐 Neutral", neu)
    c3.metric("😡 Negative", neg)

    trust, risk, trend = ai_decision(pos, neg, neu)

    # ===== TABS =====
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Dashboard", "📈 Graph Analysis", "📝 Live Data", "🧠 AI Insights"]
    )

    # ----- TAB 1 -----
    with tab1:
        pie = px.pie(df, names="Sentiment", title="Public Opinion Distribution")
        st.plotly_chart(pie, use_container_width=True)

    # ----- TAB 2 -----
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            bar = px.bar(df, x="Sentiment", title="Sentiment Count Analysis")
            st.plotly_chart(bar, use_container_width=True)

        with col2:
            df2 = df.copy()
            df2["Index"] = range(len(df2))
            line = px.line(
                df2,
                x="Index",
                y="Confidence",
                title="Sentiment Confidence Trend"
            )
            st.plotly_chart(line, use_container_width=True)

    # ----- TAB 3 -----
    with tab3:
        st.subheader("Live Hybrid Data Stream")
        st.dataframe(df, use_container_width=True)

    # ----- TAB 4 -----
    with tab4:
        st.subheader("AI Decision Intelligence System")
        st.write(f"Trust Score: {trust}")
        st.write(f"Risk Level: {risk}")
        st.write(f"Trend Direction: {trend}")
        st.write("Data Pipeline: Hybrid Live Data Engine")
        st.write("Sources: Web + Reddit")
        st.write("AI Engine: Active")
        st.write("NLP Engine: Running")
        st.write("System Status: Stable")

    # ===== AUTO REFRESH =====
    if auto_refresh:
        time.sleep(8)
        st.rerun()

else:
    st.info("👈 Enter keyword and click **Start Analysis**")
