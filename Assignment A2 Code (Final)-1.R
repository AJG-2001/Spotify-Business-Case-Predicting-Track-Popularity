###########################################################################
############################## TEAM 6 #####################################
### Assignment A2: Spotify Business Case Presentation (R Script) ##########
### Goal: Predict hit tracks & forecast track popularity trend over time ##
###########################################################################

# --- Load Required Libraries ---
library(readr)
library(dplyr)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(randomForest)
library(neuralnet)
library(caret)
library(caTools)
library(tseries)
library(forecast)
library(rugarch)
library(lubridate)
library(stringr)
library(tidyverse)
library(corrplot)
library(reshape2)

# --------------------------------|
# STEP 1: LOAD AND MERGE DATASETS |
# --------------------------------|

df_tracks <- read_csv("tracks.csv")
df_artists <- read_csv("artists.csv")

# --- Clean and join artists data to tracks ---
df_tracks$artist_id <- gsub("\\[|'|\\]", "", df_tracks$id_artists)
df_tracks$artist_id <- sub(",.*", "", df_tracks$artist_id)

df_artists_small <- df_artists[, c("id", "followers", "popularity", "genres", "name")]
colnames(df_artists_small) <- c("artist_id", "artist_followers", "artist_popularity", "artist_genres", "artist_name")

spotify_raw <- merge(df_tracks, df_artists_small, by = "artist_id", all.x = TRUE)

# --------------------------------------------|
# STEP 2: DATE FORMATTING AND TARGET VARIABLE |
# --------------------------------------------|

# Extract year, month, and year_month from release_date
spotify_raw$release_date <- as.character(spotify_raw$release_date)
spotify_raw$release_date_parsed <- parse_date_time(spotify_raw$release_date, orders = c("ymd", "ym", "y"))
spotify_raw$year <- year(spotify_raw$release_date_parsed)
spotify_raw$month <- month(spotify_raw$release_date_parsed)
spotify_raw$year_month <- format(spotify_raw$release_date_parsed, "%Y-%m")

# Create binary target: hit song if popularity > 75th percentile
pop_75 <- quantile(spotify_raw$popularity, probs = 0.75, na.rm = TRUE)
spotify_raw$businessoutcome <- as.numeric(spotify_raw$popularity > pop_75)

# ----------------------------------------|
# STEP 3: DATA CLEANING AND NORMALIZATION |
# ----------------------------------------|

spotify_clean <- spotify_raw %>%
  filter(!is.na(year), !is.na(popularity), !is.na(artist_followers),
         !is.na(artist_popularity), year >= 2000 & year <= 2023)

# Normalize input features to align scales (important for NN)
rescale <- function(x) (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))

spotify_clean <- spotify_clean %>%
  mutate(across(c(danceability, energy, valence, tempo, acousticness, instrumentalness,
                  speechiness, liveness, duration_ms, artist_followers, artist_popularity),
                ~ rescale(.), .names = "{.col}_norm"))

# -------------------------|
# STEP 4: TRAIN-TEST SPLIT |
# -------------------------|
set.seed(123)
indx <- sample(x = 1:nrow(spotify_clean), size = 0.8 * nrow(spotify_clean))
spotify_train <- spotify_clean[indx, ]
spotify_test  <- spotify_clean[-indx, ]

# -------------------------------------------------|
# STEP 5: PREDICTIVE MODELS TO CLASSIFY HIT TRACKS |
# -------------------------------------------------|

### (1.1) Gini Decision Tree [CHALLENGER MODEL]
my_tree <- rpart(businessoutcome ~ danceability + energy + valence + tempo +
                   acousticness + instrumentalness + speechiness + liveness +
                   duration_ms + artist_followers + artist_popularity,
                 data = spotify_train, method = "class", cp = 0.001)
rpart.plot(my_tree)

tree_pred <- predict(my_tree, spotify_test)
confusionMatrix(
  data = factor(as.numeric(tree_pred[, 2] > 0.5), levels = c(0, 1)),
  reference = factor(as.numeric(spotify_test$businessoutcome), levels = c(0, 1))
)

### (1.2) Gini Decision Tree (Using normalized Variables)
my_tree_norm <- rpart(
  businessoutcome ~ danceability_norm + energy_norm + valence_norm + tempo_norm +
    acousticness_norm + instrumentalness_norm + speechiness_norm + liveness_norm +
    duration_ms_norm + artist_followers_norm + artist_popularity_norm,
  data = spotify_train,
  method = "class",
  cp = 0.001
)

rpart.plot(my_tree_norm)

tree_pred_norm <- predict(my_tree_norm, spotify_test)

confusionMatrix(
  data = factor(as.numeric(tree_pred_norm[, 2] > 0.5), levels = c(0, 1)),
  reference = factor(as.numeric(spotify_test$businessoutcome), levels = c(0, 1))
)

### (2.1) Random Forest (adds ensemble strength and variable importance) [CHAMPION MODEL]
forest_model <- randomForest(
  x = spotify_train[, c('danceability','energy','valence','tempo',
                        'acousticness','instrumentalness','speechiness',
                        'liveness','duration_ms','artist_followers','artist_popularity')],
  y = as.factor(spotify_train$businessoutcome),
  ntree = 500
)
forest_pred <- predict(forest_model, spotify_test)
confusionMatrix(forest_pred, as.factor(spotify_test$businessoutcome))
varImpPlot(forest_model)  # artist_popularity typically ranks high

### (2.2) Random Forest (Using normalized Variables) 
forest_model_norm <- randomForest(
  x = spotify_train[, c(
    'danceability_norm', 'energy_norm', 'valence_norm', 'tempo_norm',
    'acousticness_norm', 'instrumentalness_norm', 'speechiness_norm',
    'liveness_norm', 'duration_ms_norm', 'artist_followers_norm', 'artist_popularity_norm'
  )],
  y = as.factor(spotify_train$businessoutcome),
  ntree = 500
)

forest_pred_norm <- predict(forest_model_norm, spotify_test)

confusionMatrix(
  data = factor(forest_pred_norm, levels = c(0, 1)),
  reference = factor(spotify_test$businessoutcome, levels = c(0, 1))
)

### (3.1) Neural Network (captures nonlinear patterns)
my_neural <- neuralnet(businessoutcome ~ danceability + energy + valence + tempo +
                         acousticness + instrumentalness + speechiness + liveness +
                         duration_ms + artist_followers + artist_popularity,
                       data = spotify_train, hidden = c(4, 2), linear.output = FALSE)
plot(my_neural, rep = "best")

neural_pred <- predict(my_neural, spotify_test)
confusionMatrix(
  data = factor(as.numeric(neural_pred > 0.5), levels = c(0, 1)),
  reference = factor(as.numeric(spotify_test$businessoutcome), levels = c(0, 1))
)

### (3.2) Neural Network (Using normalized Variables)[WARNING!!!: TAKES TOO LONG TO RUN]
my_neural_norm <- neuralnet(
  businessoutcome ~ danceability_norm + energy_norm + valence_norm + tempo_norm +
    acousticness_norm + instrumentalness_norm + speechiness_norm + liveness_norm +
    duration_ms_norm + artist_followers_norm + artist_popularity_norm,
  data = spotify_train,
  hidden = c(3, 2),
  linear.output = FALSE
)

plot(my_neural_norm, rep = "best")

neural_pred_norm <- predict(my_neural_norm, spotify_test)

confusionMatrix(
  data = factor(as.numeric(neural_pred_norm > 0.5), levels = c(0, 1)),
  reference = factor(as.numeric(spotify_test$businessoutcome), levels = c(0, 1))
)

### (4.1) Logistic Regression (baseline linear model)
my_logit <- glm(businessoutcome ~ danceability + energy + valence + tempo +
                  acousticness + instrumentalness + speechiness + liveness +
                  duration_ms + artist_followers + artist_popularity,
                data = spotify_train, family = "binomial")

logit_pred <- predict(my_logit, spotify_test)
confusionMatrix(
  data = factor(as.numeric(logit_pred > 0.5), levels = c(0, 1)),
  reference = factor(as.numeric(spotify_test$businessoutcome), levels = c(0, 1))
)

### (4.2) Logistic Regression (Using normalized Variables)
my_logit_norm <- glm(
  businessoutcome ~ danceability_norm + energy_norm + valence_norm + tempo_norm +
    acousticness_norm + instrumentalness_norm + speechiness_norm + liveness_norm +
    duration_ms_norm + artist_followers_norm + artist_popularity_norm,
  data = spotify_train,
  family = "binomial"
)

logit_pred_norm <- predict(my_logit_norm, spotify_test)

confusionMatrix(
  data = factor(as.numeric(logit_pred_norm > 0.5), levels = c(0, 1)),
  reference = factor(as.numeric(spotify_test$businessoutcome), levels = c(0, 1))
)

# ----------------------------------------|
# STEP 6: Descriptive & Exploratory Plots |
# ----------------------------------------|

### Distribution of Track Popularity
ggplot(spotify_clean, aes(x = popularity)) +
  geom_histogram(binwidth = 5, fill = "steelblue", color = "white", alpha = 0.7) +
  geom_density(aes(y = ..density.. * 5), color = "red", size = 1) +
  labs(title = "Distribution of Track Popularity",
       x = "Popularity Score", y = "Count / Density") +
  theme_minimal()

### Correlation Heatmap
num_vars <- spotify_clean %>%
  select(danceability, energy, valence, tempo, acousticness, instrumentalness,
         speechiness, liveness, duration_ms, artist_followers, artist_popularity, popularity)
corr_matrix <- cor(num_vars, use = "complete.obs")
corrplot(corr_matrix, method = "color", type = "upper", tl.cex = 0.8)

### Track Popularity by Year
ggplot(spotify_clean, aes(x = as.factor(year), y = popularity)) +
  geom_boxplot(fill = "lightgreen", outlier.color = "red", outlier.size = 1) +
  labs(title = "Track Popularity by Year",
       x = "Year", y = "Popularity") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

### Hit Rate by Genre (Top 10)
top_genres <- spotify_clean %>%
  filter(!is.na(artist_genres)) %>%
  count(artist_genres, sort = TRUE) %>%
  slice(1:10) %>%
  pull(artist_genres)
hit_rate <- spotify_clean %>%
  filter(artist_genres %in% top_genres) %>%
  group_by(artist_genres) %>%
  summarise(hit_rate = mean(businessoutcome, na.rm = TRUE))
ggplot(hit_rate, aes(x = reorder(artist_genres, hit_rate), y = hit_rate)) +
  geom_col(fill = "dodgerblue") +
  coord_flip() +
  labs(title = "Hit Rate by Genre (Top 10 Genres)",
       x = "Genre", y = "Proportion of Hit Songs") +
  theme_minimal()

### Confusion Matrix Visualization (Random Forest)
cm <- confusionMatrix(forest_pred, as.factor(spotify_test$businessoutcome))
cm_table <- as.table(cm$table)
cm_melt <- melt(cm_table)
ggplot(cm_melt, aes(Reference, Prediction, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), color = "white", size = 6) +
  scale_fill_gradient(low = "skyblue", high = "navy") +
  labs(title = "Confusion Matrix: Random Forest", x = "Actual", y = "Predicted") +
  theme_minimal()

### Distribution of Key Audio Features
audio_features <- c("danceability", "energy", "valence", "acousticness", "instrumentalness", "speechiness", "liveness")
spotify_long <- spotify_clean %>%
  select(all_of(audio_features)) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "value")
ggplot(spotify_long, aes(x = value, fill = feature)) +
  geom_histogram(bins = 30, alpha = 0.6, position = "identity") +
  facet_wrap(~feature, scales = "free") +
  labs(title = "Distribution of Key Audio Features") +
  theme_minimal()

### Monthly average popularity for Top 5 Genres
top_genres <- spotify_clean %>%
  filter(!is.na(artist_genres)) %>%
  count(artist_genres, sort = TRUE) %>%
  slice(1:5) %>%
  pull(artist_genres)

genre_trends <- spotify_clean %>%
  filter(artist_genres %in% top_genres) %>%
  group_by(year_month, artist_genres) %>%
  summarise(avg_popularity = mean(popularity, na.rm = TRUE), .groups = "drop")
genre_trends$year_month <- as.Date(paste0(genre_trends$year_month, "-01"))

ggplot(genre_trends, aes(x = year_month, y = avg_popularity, color = artist_genres)) +
  geom_line(size = 1) +
  labs(title = "Monthly Track Popularity for Top 5 Genres",
       x = "Year", y = "Average Popularity", color = "Genre") +
  theme_minimal()

### Monthly average popularity for Top 5 Artists
top_artists <- spotify_clean %>%
  filter(!is.na(artist_name)) %>%
  count(artist_name, sort = TRUE) %>%
  slice(1:5) %>%
  pull(artist_name)

artist_trends <- spotify_clean %>%
  filter(artist_name %in% top_artists) %>%
  group_by(year_month, artist_name) %>%
  summarise(avg_popularity = mean(popularity), .groups = "drop")
artist_trends$year_month <- as.Date(paste0(artist_trends$year_month, "-01"))

ggplot(artist_trends, aes(x = year_month, y = avg_popularity, color = artist_name)) +
  geom_line(size = 1) +
  labs(title = "Monthly Track Popularity for Top 5 Artists",
       x = "Year", y = "Average Popularity", color = "Artist") +
  theme_minimal()

# -------------------------------------------------------------------|
# STEP 7: TIME SERIES FORECASTING (ARIMA) – OVERALL TRACK POPULARITY |
# -------------------------------------------------------------------|

### Business Objective: Understand how listening trends are changing
monthly_popularity <- spotify_clean %>%
  group_by(year_month) %>%
  summarise(avg_popularity = mean(popularity, na.rm = TRUE)) %>%
  arrange(year_month)

popularity_ts <- ts(monthly_popularity$avg_popularity,
                    start = c(2000, 1), frequency = 12)

plot(popularity_ts, main = "Average Monthly Track Popularity",
     ylab = "Popularity", xlab = "Year")

### Stationarity and diagnostics
adf.test(popularity_ts)
acf(popularity_ts)
pacf(popularity_ts)

### Fit ARIMA
auto_model <- auto.arima(popularity_ts)
summary(auto_model)

### Forecast
forecast_pop <- forecast(auto_model, h = 12)
plot(forecast_pop,
     main = "12-Month Forecast of Track Popularity",
     ylab = "Popularity", xlab = "Year")

