# **Spotify Business Case – Predicting Track Popularity**

📌 Project Overview

This project was developed as part of the Business Case assignment, where the objective was to analyze Spotify’s track dataset and determine what makes a song popular.

The team used R for data analysis and modeling and created both a forecasting model and classification models to deliver business insights and actionable recommendations. Results were presented through a 6-minute YouTube video, an R code submission, and a presentation deck
.

🔎 Dataset

Source: Spotify Tracks Dataset – Kaggle

Scope: Millions of tracks with metadata and popularity scores.

Key Features:

Artist popularity

Followers

Duration (ms)

Danceability, acousticness, liveness, valence, speechiness

Popularity (target variable for classification/forecasting)

🛠️ Methodology
Classification Models (Hit Prediction)

Definition of Hit: Tracks in the top 25% popularity scores.

Models Applied:

GINI Decision Tree – Baseline challenger.

Random Forest – Champion model (best accuracy).

Neural Network – Captured non-linear relations, weaker overall.

Logistic Regression – Solid baseline, strong sensitivity but more false positives.

Best Performer: Random Forest with 77% accuracy, leveraging artist popularity, followers, and danceability as the strongest predictors
.

Forecasting Model (Popularity Trends)

Model Used: ARIMA(1,0,1)(1,1,0)[12]

Diagnostics:

ADF test confirmed stationarity (p < 0.05).

Strong seasonal cycle at lag 12 (yearly).

Forecast Results:

Steady growth expected (popularity rising to 48–52 points).

December spikes (holiday effect) and Feb/March troughs observed.

Confidence intervals widen in Q4, indicating higher volatility
.

📊 Business Insights

Hit Prediction

Social/artist metrics (followers, popularity) outperform audio features.

Random Forest enables precise targeting of promotional campaigns.

Seasonality & Forecasting

Popularity peaks in December → optimize Q4 releases.

Q2–Q3 show stable growth → ideal for mid-year marketing pushes.

February–March dips → reduce marketing spend or buffer budgets.

Strategic Recommendations

Automate retraining of models quarterly to refine predictions.

Expand feature set with social media and lyrics sentiment for improved accuracy.

Develop genre- and region-specific models for tailored strategies.

📂 Deliverables

Team 6 - A2 PPT.pdf – Presentation slides summarizing methods and findings.

R Code (not included here, but submitted separately for grading integrity).

YouTube Video – 6-minute presentation explaining insights, models, and recommendations.

🚀 Key Takeaways

Random Forest is the most effective classifier for hit prediction.

ARIMA forecasting captured seasonality and momentum, guiding release calendar planning.

Artist popularity and followers are the most decisive predictors of success.

Actionable strategies include holiday-timed releases, budget optimization, and social media integration
.
