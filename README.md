# Fake News Detection with PySpark

This notebook implements an end-to-end machine learning pipeline for detecting fake news articles using Apache Spark. It combines text processing, metadata features, and multiple classification models to identify misleading or false content.

## Dataset

The project uses a [Fake News Detection dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) from Kaggle, consisting of two files:

- `True.csv`: Real news articles
- `Fake.csv`: Fake news articles from sources flagged by PolitiFact and Wikipedia

Each article includes the title, main text, subject, and publication date. Articles are primarily political and published around 2016–2017.

## Pipeline Overview

The notebook follows a multi-stage Spark ML pipeline:

1. **Text Preprocessing**
   - Tokenization using `RegexTokenizer`
   - Custom stopword removal
   - TF-IDF vectorization (top 5,000 terms)

2. **Metadata Feature Engineering**
   - Features such as exclamation and ALL CAPS counts, date fields, and text length
   - Missing date and text imputation with explicit flags

3. **Model Training**
   - Logistic Regression (baseline and tuned)
   - Random Forest

4. **Model Evaluation**
   - Accuracy, precision, recall, AUC-ROC, and AUC-PR
   - Feature importance analysis (metadata and TF-IDF)

## Results

**Best-performing model**: Tuned Logistic Regression  
- **Accuracy**: 98.9%  
- **Precision (fake = 1)**: 99.3%  
- **Recall (fake = 1)**: 98.5%  
- **AUC-ROC**: 0.9988  

The tuned logistic regression model achieved the best balance between precision and recall, outperforming random forests, which had higher precision but much lower recall.

### Key Findings

- **Stylistic features** (like `all_caps_count` and `exclam_count`) were strong indicators of fake news.
- **Missing or ambiguous metadata** (e.g., unparseable dates, 2016 election references) correlated with fake articles.
- TF-IDF terms like `washington`, `corrects`, and day-of-week references were among the most informative text features.

This notebook demonstrates how scalable tools like PySpark can be used for realistic, interpretable fake news detection pipelines combining NLP and metadata.
