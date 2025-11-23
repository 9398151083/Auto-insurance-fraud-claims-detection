📘 Automated Insurance Claim Fraud Detection System

This project builds an **end-to-end AI system** that analyzes auto-insurance claims, detects potentially fraudulent submissions, and prioritizes them for investigator review.

The system mimics real-world insurance fraud workflows and includes:

* Anomaly Detection
* Supervised ML Models
* Network-Graph Fraud Ring Detection
* Combined Risk Scoring
* Investigation Queue with Priority Ranking
* Visualizations & Model Evaluation

---

## 🚀 Key Features

✔ **Flags suspicious claims**
✔ **Handles missing documents & messy data**
✔ **Detects organized fraud (fraud rings)**
✔ **Uses multiple ML models + anomaly detection**
✔ **Creates an investigation queue with priority rank**
✔ **Confidence scoring for investigator trust**
✔ **Visualizations for EDA & graph insights**

---

## 📂 Project Structure

```
project/
│── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── network_analysis.py
│   ├── models.py
│   ├── scoring.py
│   ├── pipeline.py
│   └── predict.py
│
│── notebooks/
│   └── Fraud_Detection_Notebook.ipynb   # Full interactive pipeline + visualizations
│
│── investigation_queue.csv              # Generated system output
│── requirements.txt
│── README.md
```

---

# 🛠️ Step-by-Step: What the System Does

Below is a clear explanation you can speak in an interview.

---

## **1. Data Ingestion**

* Load `insurance_claims.csv`
* Clean column names
* Convert numbers, dates
* Handle missing documentation using `"UNKNOWN"`

**Why:** Insurance datasets are messy; robust ingestion prevents pipeline failures.

---

## **2. Exploratory Data Analysis (EDA)**

The notebook shows:

* Missing value heatmap
* Claim amount distribution
* Fraud vs non-fraud ratio
* Category count plots
* Correlation heatmap

**Why:** Understand fraud patterns, skewed distributions, and data leakage risk.

---

## **3. Feature Engineering**

Includes:

* Log of claim amount
* Claim ratio features:

  * injury_claim / total
  * property_claim / total
  * vehicle_claim / total
* Date features (month, weekday)
* Top categorical encodings
* Binary flags for missing documents

**Why:** Fraud patterns show up in ratios, time trends, and repeated categories.

---

## **4. Network Analysis (Fraud Rings Detection)**

Graph nodes created for:

* Claim
* Policy
* Customer
* Auto Make / Manufacturer
* City
* Zipcode
* Incident Type
* Collision Type
* Injury Severity

We connect them to detect:

* Reused policies
* Shared addresses
* Same car make models in multiple suspicious incidents
* Dense clusters → potential fraud rings

A **network_score** is computed from node degree and cluster density.

**Why:** Fraud rings are impossible to catch from rows — graph reveals patterns.

---

## **5. Machine Learning Models**

### **Unsupervised Model**

✔ **Isolation Forest** → detects unusual claims
Good for:

* New fraud patterns
* Previously unseen cases

### **Supervised Models**

* Random Forest
* Logistic Regression
* Gradient Boosting
* XGBoost (optional)

Model selection based on **validation ROC AUC**.

### **Handling Class Imbalance**

* Uses **SMOTE** to oversample fraud in training data
* Prevents bias toward predicting “Not Fraud”

---

## **6. Combined Risk Scoring**

We combine 3 signals:

| Component       | Meaning                             |
| --------------- | ----------------------------------- |
| `anomaly_score` | how unusual the claim is            |
| `clf_proba`     | probability predicted by classifier |
| `network_score` | graph-based fraud ring score        |

### **Risk Score Formula**

```
risk = 0.40 * anomaly + 0.45 * classifier + 0.15 * network
```

### **Confidence Score**

```
confidence = 1 - |anomaly - classifier|
```

Higher confidence = both models agree.

---

## **7. Generate Investigation Queue (Final Output)**

The system creates:

### `investigation_queue.csv`

Columns include:

* `risk_score`
* `confidence`
* `fraud_flag` (1 = investigate)
* `action` (INVESTIGATE or AUTO-PAY)
* `priority_rank`

### **Priority Ranking Logic**

Sorted by:

1. Highest risk
2. Highest confidence
3. **Higher claim amount (tie-breaker)**

This ensures **expensive claims get reviewed first** if equally suspicious.

---

## **8. Visualizations**

Included in the notebook:

📊 Claim Amount Distribution
📊 Missing Data Heatmap
📊 Fraud Correlation Heatmap
📊 Feature Importance Plot
📊 Risk Score Histogram
📊 ROC Curve
🕸️ Fraud Ring Network Graph Visualization

---

# 🧪 How to Run

### **Step 1 — Install dependencies**

```
pip install -r requirements.txt
```

### **Step 2 — Run the full pipeline**

```
python src/pipeline.py --csv "path/to/insurance_claims.csv"
```

### **Step 3 — View results**

```
investigation_queue.csv
```

Or open the notebook:

### **Run notebook version (recommended for interview)**

```
notebooks/Fraud_Detection_Notebook.ipynb
```

---

# 🎯 Investigate vs Auto-Pay Logic

The pipeline automatically assigns:

| Condition              | Action      |
| ---------------------- | ----------- |
| risk_score ≥ threshold | INVESTIGATE |
| else                   | AUTO-PAY    |

Threshold is adjustable:

* Fixed (0.75)
* Percentile (Top 5% highest risk)
* Statistical (mean + std)

---

# 📈 Evaluation

The project calculates:

* ROC AUC
* Confusion matrix
* Precision / Recall
* Feature importance
* Fraud ring graph density

The notebook contains visualizations for all metrics.

---

# 🧩 Why This Design Is Excellent for Interviews

Your solution shows:

✔ Multi-model fraud detection (anomaly + supervised)
✔ Graph-based fraud ring detection (very impressive!)
✔ Proper ML pipeline engineering
✔ Investigation prioritization (business requirement)
✔ Realistic insurance fraud handling
✔ Clean modular code following best practices
✔ Professional notebook with visualizations

This aligns **exactly** with the skills organizations want:

* AI/ML
* Critical reasoning
* Fraud domain knowledge
* System design
* Modular engineering

---

# 🏁 Conclusion

This system:

✔ Flags the fraud
✔ Builds an investigation queue
✔ Computes priority
✔ Applies confidence scoring
✔ Detects fraud rings
✔ Provides visual insights
✔ Delivers a full ML pipeline





