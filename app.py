import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import io

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# -----------------
# 1️⃣ Load Data (cached for speed)
# -----------------
@st.cache_data
def load_data():
    return pd.read_excel("political sentiment new data.xlsx", engine='openpyxl')

df = load_data()

# -----------------
# 2️⃣ Global color map
# -----------------
color_map = {
    'Positive': 'green',
    'Neutral': 'gray',
    'Negative': 'red',
    'BJP': '#FF9933',
    'Congress': '#138808',
    'AAP': '#1A73E8'
}

# Fix sentiment score mapping
df['sentiment_score'] = df['sentiment_label'].map({
    'Positive': 0.5,
    'Neutral': 0.0,
    'Negative': -0.5
})

# Preprocess
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['month'] = df['timestamp'].dt.to_period('M').astype(str)

# -----------------
# 3️⃣ Precompute Encoders BEFORE filtering
# -----------------
party_codes = df['party'].astype('category').cat.categories
party_decoder = {p: i for i, p in enumerate(party_codes)}
location_codes = df['location'].astype('category').cat.categories
location_decoder = {l: i for i, l in enumerate(location_codes)}

# -----------------
# 4️⃣ Sidebar Navigation
# -----------------
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to Section:", [
    "Overview", "Donut Chart", "Sentiment Bars", "Time Series",
    "Polarized Cities", "Leaderboard", "Net Sentiment", "ML Model"
])

# Filters
st.sidebar.header("🔎 Filters")
selected_parties = st.sidebar.multiselect("Select Parties:", df['party'].unique(), default=df['party'].unique())
selected_locations = st.sidebar.multiselect("Select Cities:", df['location'].unique(), default=df['location'].unique())
selected_sentiments = st.sidebar.multiselect("Select Sentiments:", df['sentiment_label'].unique(), default=df['sentiment_label'].unique())

# Apply filters
filtered_df = df[
    (df['party'].isin(selected_parties)) &
    (df['location'].isin(selected_locations)) &
    (df['sentiment_label'].isin(selected_sentiments))
]

# -----------------
# 5️⃣ Title
# -----------------
st.markdown("""
    <h1 style='text-align: center; color: #444;'>🗳️ <span style='color:#FF9933'>Political</span> <span style='color:#138808'>Sentiment</span> <span style='color:#1A73E8'>Dashboard</span> <br> <span style='font-size: 18px;'>June 2024 – June 2025</span></h1>
""", unsafe_allow_html=True)

# -----------------
# Overview
# -----------------
if page == "Overview":
    st.subheader("📊 Summary Statistics")
    st.metric("Total Records", len(filtered_df))
    st.metric("Average Sentiment Score", round(filtered_df['sentiment_score'].mean(), 2))

    st.subheader("🏳️ Party Legend")
    legend_cols = st.columns(3)
    for col, (party, img) in zip(legend_cols, [
        ("BJP", "bjp.png"), ("Congress", "congress.png"), ("AAP", "aap.png")
    ]):
        with col:
            st.image(img, width=60)
            st.markdown(f"<center><b style='color:{color_map[party]}'>{party}</b></center>", unsafe_allow_html=True)

    st.subheader("📈 Monthly KPI Trends")
    monthly_df = filtered_df.copy()
    monthly_df['month'] = monthly_df['timestamp'].dt.to_period('M').astype(str)
    monthly_sentiment_avg = monthly_df.groupby('month')['sentiment_score'].mean()
    fig_avg = px.line(monthly_sentiment_avg.reset_index(), x='month', y='sentiment_score',
                      title="📊 Average Sentiment Score Over Time", markers=True,
                      template="plotly_white")
    st.plotly_chart(fig_avg)
    st.download_button("📥 Download Chart (PNG)", data=pio.to_image(fig_avg, format='png'),
                       file_name="average_sentiment.png", mime="image/png")

# -----------------
# Donut Chart
# -----------------
elif page == "Donut Chart":
    st.subheader("🍩 Party-Wise Sentiment Distribution")
    parties = list(filtered_df['party'].unique())  # make sure it's a list of strings
    tab_objs = st.tabs(parties)

    for tab, party in zip(tab_objs, parties):
        with tab:
            party_df = filtered_df[filtered_df['party'] == party]
            fig_donut = px.pie(party_df, names='sentiment_label', color='sentiment_label',
                               color_discrete_map=color_map, hole=0.4,
                               title=f"{party} Sentiment Distribution")
            st.plotly_chart(fig_donut, use_container_width=True)

# -----------------
# Sentiment Bars
# -----------------
elif page == "Sentiment Bars":
    st.subheader("🎯 Sentiment Distribution by Party")
    sentiment_counts = filtered_df.groupby(['party', 'sentiment_label']).size().reset_index(name='count')
    fig_sentiment = px.bar(sentiment_counts, x='party', y='count', color='sentiment_label',
                           barmode='group', title="Sentiment Count per Party",
                           color_discrete_map=color_map)
    st.plotly_chart(fig_sentiment)

# -----------------
# Time Series
# -----------------
elif page == "Time Series":
    st.subheader("📈 Sentiment Over Time")
    time_series = filtered_df.groupby(['month', 'sentiment_label']).size().reset_index(name='count')
    fig_trend = px.line(time_series, x='month', y='count', color='sentiment_label', markers=True,
                        title="Sentiment Trend Over Time",
                        color_discrete_map=color_map,
                        template="plotly_white")
    st.plotly_chart(fig_trend)

# -----------------
# Polarized Cities (Most Polarized)
# -----------------
elif page == "Polarized Cities":
    st.subheader("⚖️ Top 5 Most Polarized Cities")
    st.caption("Polarity Gap = |Positive% − Negative%|")
    city_counts = filtered_df.groupby(['location', 'sentiment_label']).size().unstack(fill_value=0)
    city_counts['Total'] = city_counts.sum(axis=1)
    city_counts = city_counts[city_counts['Total'] >= 100]
    city_counts['Polarity Gap'] = abs(city_counts['Positive']/city_counts['Total']*100 - city_counts['Negative']/city_counts['Total']*100)
    polarized = city_counts.sort_values(by='Polarity Gap', ascending=False).head(5).reset_index()
    st.dataframe(polarized[['location', 'Polarity Gap', 'Total']].round(2))

# -----------------
# Leaderboard
# -----------------
elif page == "Leaderboard":
    st.subheader("🏆 Sentiment Leaderboard by Party")
    avg_city_scores = filtered_df.groupby(['party', 'location'])['sentiment_score'].mean().reset_index()
    leaderboard_rows = []
    for party in avg_city_scores['party'].unique():
        party_df = avg_city_scores[avg_city_scores['party'] == party]
        top_city = party_df.sort_values(by='sentiment_score', ascending=False).iloc[0]['location']
        bottom_city = party_df.sort_values(by='sentiment_score', ascending=True).iloc[0]['location']
        leaderboard_rows.append({"Party": party, "Happiest City": top_city, "Angriest City": bottom_city})
    st.dataframe(pd.DataFrame(leaderboard_rows))

# -----------------
# Net Sentiment
# -----------------
elif page == "Net Sentiment":
    st.subheader("🏙️ Net Sentiment by City & Party")
    grouped = filtered_df.groupby(['location', 'party', 'sentiment_label']).size().unstack(fill_value=0)
    grouped['Total'] = grouped.sum(axis=1)
    grouped['Net Sentiment'] = ((grouped['Positive'] - grouped['Negative']) / grouped['Total'] * 100).round(2)
    heatmap_data = grouped.reset_index().pivot(index='location', columns='party', values='Net Sentiment').fillna(0).round(2)
    st.dataframe(heatmap_data.style.background_gradient(cmap='RdYlGn', axis=1))

# -----------------
# ML Model (Precision only)
# -----------------
elif page == "ML Model":
    st.subheader("🤖 ML Model: Predict Sentiment Label")
    try:
        df_ml = df.copy()
        df_ml['month_num'] = pd.to_datetime(df_ml['timestamp']).dt.month
        df_ml['party_encoded'] = df_ml['party'].map(party_decoder)
        df_ml['location_encoded'] = df_ml['location'].map(location_decoder)

        X = df_ml[['party_encoded', 'location_encoded', 'month_num']]
        y = df_ml['sentiment_label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # Precision only report
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        precision_lines = []
        precisions = {}
        for label in ['Negative', 'Neutral', 'Positive']:
            if label in report_dict:
                precision_val = report_dict[label]['precision']
                precisions[label] = precision_val
                precision_lines.append(f"{label:<10} Precision: {precision_val:.2f}")
        st.code("\n".join(precision_lines))

        # Optional: Horizontal bar chart for precision
        fig_precision = px.bar(
            x=list(precisions.values()),
            y=list(precisions.keys()),
            orientation='h',
            title="Model Precision by Class",
            text=[f"{v:.2f}" for v in precisions.values()],
            range_x=[0, 1]
        )
        st.plotly_chart(fig_precision)

        st.markdown("---")
        st.subheader("🔮 Try It Yourself – Predict Sentiment")
        party_input = st.selectbox("Select Party:", options=list(party_decoder.keys()))
        location_input = st.selectbox("Select City:", options=list(location_decoder.keys()))
        month_input = st.selectbox("Select Month (1–12):", options=list(range(1, 13)))

        if st.button("Predict Sentiment"):
            input_features = [[
                party_decoder[party_input],
                location_decoder[location_input],
                month_input
            ]]
            prediction = clf.predict(input_features)[0]
            confidence = clf.predict_proba(input_features).max() * 100
            st.success(f"📢 Predicted Sentiment: **{prediction}** ({confidence:.2f}% confidence)")

    except Exception as e:
        st.error(f"Model training or prediction failed: {e}")
