# Uncovering Business Success Factors Through Machine Learning on Yelp Data
 ---
## Data Cleaning  
#### 00a_cleaning_script.ipynb
- Processes data after initial OpenRefine cleaning  
- Filters reviews to only include restaurants from cleaned business data  
- Performs additional cleanup: column renaming, value standardization  
- Outputs processed data for further OpenRefine refinement  
 ---
## DistilBERT Sentiment Classification Pipeline
#### 00b_EDA_Yelp_Reviews.ipynb
- Loads and cleans Yelp review data, including basic text preprocessing and feature engineering.
- Performs exploratory data analysis on review metadata (e.g., review length, star ratings, engagement indicators).
- Informs the design of contextual tokens used in fine-tuning.

#### 00c_WorkingData_Setup.ipynb
- Merges Yelp review, user, and business datasets to construct a unified modeling dataset.
- Generates derived features (e.g., rating stability, business maturity) and performs fuzzy matching on city names.
- Outputs a processed dataset (`reviews_working.csv`) for use in sentiment classification wwith fine-tuned model.

#### 01_Baseline_DistilBERT.ipynb
- Implements pre-trained DistilBERT for sentiment classification on pre-/post-covid segments of Yelp reviews.
- Uses SHAP to explain predictions on review subsets

#### 02a_FineTune_DistilBERT.ipynb
- Fine-tunes a pre-trained DistilBERT model from Hugging Face on a stratified sample of Yelp reviews.
- Augments text with structured metadata via special tokens to provide additional context.
- Implements class-weighted training using Hugging Face’s Trainer API and evaluates performance on a held-out test set.

#### 02b_Apply_FinetunedDB_FullYelp.ipynb
- Applies the fine-tuned DistilBERT model to the full Yelp dataset for sentiment inference.
- Compares results to baseline predictions and highlights reviews with the largest discrepancies.
- Generates TF-IDF-based word clouds and extracts distinctive textual themes from key review segments.

---
## 03_Clustering  
#### clustering.ipynb
- Applies K-Means to group restaurants using price_range/review_count  
- Optimizes clusters via silhouette scores and elbow method  
 ---
## 04_Classification
#### classification.ipynb
- Applies CatBoost model to determine feature importance for model
- Optimizes via gradient boosted trees, which is evaluated through classification matrix and metrics such as accuracy, precision, recall, F1 score
 ---
 
*Data Note: Raw Yelp files used (business.json, review.json, user.json) can be downloaded from [Yelp Open Dataset](https://www.yelp.com/dataset). Raw and cleaned datasets are also excluded from the repo due to size constraints.*  
 
 
