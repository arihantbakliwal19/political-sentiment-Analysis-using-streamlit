# 🗳️ Political Sentiment Analysis using Streamlit


This project is an interactive data analytics and machine learning dashboard that tracks and analyzes political sentiment across multiple parties and cities in India.
It uses publicly collected sentiment data (June 2024 – June 2025) to visualize trends, compare party performance, and even predict sentiment labels using a trained ML model.
Built entirely in Streamlit for a clean, interactive user experience.


## 🚀 Features

Party-wise Sentiment Donut Charts – Visual breakdown of sentiment distribution for each political party.

Sentiment Trends Over Time – Interactive time-series showing sentiment shifts month-to-month.

Most Polarized Cities Analysis – Highlights cities with the widest gap between positive and negative sentiment.

Leaderboard – Shows each party’s happiest and angriest cities.

Net Sentiment Heatmap – Compares net sentiment (Positive − Negative %) by city & party.

ML Prediction Tool – Enter party, city, and month to get a predicted sentiment label with confidence.



## 📂 Project Structure

political-sentiment-dashboard-using-streamlit/
├── app.py # Main Streamlit dashboard
├── notebooks/
│ └── political sentiment.ipynb # EDA & ML model building
├── data/
│ └── political sentiment new data.xlsx # Dataset
├── images/
│ ├── bjp.png
│ ├── congress.png
│ └── aap.png
├── requirements.txt
└── README.md




## 🛠️ Technologies Used

Python

Streamlit – for dashboard and UI

Pandas – for data preprocessing and analysis

Plotly – for interactive charts

Scikit-learn – Random Forest classifier for sentiment prediction

OpenPyXL – to load Excel-based datasets



## 💻 How It Works
Data Loading – Sentiment data is loaded from an Excel file and preprocessed (timestamp conversion, sentiment scoring).

Visualization – Multiple interactive charts and tables let users filter by party, city, or sentiment type.

Polarization Analysis – Calculates polarity gap to find cities with extreme sentiment divides.

ML Model Training – A Random Forest classifier predicts sentiment labels based on party, location, and month.

Interactive Prediction – Users can input party, city, and month to get a sentiment prediction in real time.


## 📄 Example Scenario
Use Case:
You want to see which cities have the highest positive sentiment for BJP in early 2025, or compare Congress and AAP’s sentiment trends in Delhi.
You can filter directly in the sidebar and instantly view charts, tables, and net sentiment values.
ML Example:

Input: Party = AAP, City = Mumbai, Month = 3 (March)

Output: Predicted Sentiment = Neutral (Confidence: 74.5%)


## 📬 Author  

[**Arihant Bakliwal**](https://github.com/arihantbakliwal)  


