# # Problem Definition

# **Business Problem Definition:**
#
# The objective of this project is to develop a predictive model to identify individuals at risk of diabetes based on various health parameters. Early detection of diabetes risk can significantly improve patient outcomes by enabling timely interventions and lifestyle modifications. The model will use historical data on patients' health metrics such as glucose levels, insulin levels, BMI (Body Mass Index), age, and other relevant factors to predict the likelihood of an individual developing diabetes.
# The project aims to develop a reliable predictive model that can assist healthcare professionals in identifying individuals at risk of diabetes, enabling early intervention and personalized healthcare management strategies.
#
# **Project Steps with Detailed Explanations:**
#
# 1. **Data Loading and Exploration:**
#    - Load the dataset containing historical health data of individuals.
#    - Perform initial exploration to understand the structure of the data, including dimensions, data types, and missing values.
#
# 2. **Exploratory Data Analysis (EDA):**
#    - Conduct in-depth analysis of the data to gain insights into the distribution and relationships between different variables.
#    - Explore categorical variables by calculating frequencies and visualizing distributions.
#    - Analyze numerical variables by computing descriptive statistics and visualizing distributions.
#    - Investigate correlations between variables using correlation matrices and visualizations.
#
# 3. **Feature Engineering:**
#    - Identify and handle zero values in certain columns, replacing them with appropriate values such as the median.
#    - Handle missing values by imputing them using relevant statistical measures.
#    - Detect and treat outliers using suitable methods to ensure they don't unduly influence the model.
#    - Create new features that may capture important relationships or patterns in the data, such as age categories, BMI categories, and insulin-to-BMI ratio.
#    - Encode categorical variables and create dummy variables for categorical features with multiple levels.
#
# 4. **Model Development:**
#    - Split the dataset into training and testing sets to evaluate model performance.
#    - Standardize numerical features to ensure all features contribute equally to the model.
#    - Train a Random Forest Classifier, a robust machine learning algorithm capable of handling complex datasets and capturing nonlinear relationships.
#
# 5. **Model Evaluation:**
#    - Make predictions using the trained model on the test set.
#    - Evaluate model performance using metrics such as accuracy, precision, recall, F1 score, and ROC AUC score.
#    - Summarize the results in a tabular format to compare the performance of the model.
#

# ### 1. **Data Loading and Exploration:**
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("/Users/asmir/Desktop/MyProjects/DiabetPredicton/diabetes.csv", sep=',')

# ### 2. **Exploratory Data Analysis (EDA):**
#

# ####            # Step 1: Overview of the dataset.
#
#

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df, head=2)


# #### # Step 2: Capture numerical and categorical variables.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)


# ####  Step 3: Analyze numerical and categorical variables.

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)


#  #### Step 4: Analyze the target variable.

# Mean of target variable by categorical variables,
# Mean of numerical variables by target variable)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


for col in num_cols:
    num_summary(df, col, plot=True)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


# #### Step 5: Outlier analysis.

# Outlier Detection and Replacement
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) |
                 (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Outlier Detection

for col in df.columns:
    print(col, check_outlier(df, col))

    if check_outlier(df, col):
        replace_with_thresholds(df, col)

# #### Step 6: Missing data analysis.

# Handling with Missing Data
# Although there are no missing observations in the dataset,
# observations with zero values in features like Glucose and Insulin
# could indicate missing values.
# Replace zero values with NaN and handle missing values accordingly.

zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

na_columns = df.columns[df.isnull().any()]

for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

for col in df.columns:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

# #### Step 7: Correlation analysis.

df.corr()

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

diabetic = df[df.Outcome == 1]
healthy = df[df.Outcome == 0]

plt.scatter(healthy.Age, healthy.Insulin, color="green", label="Healthy", alpha=0.4)
plt.scatter(diabetic.Age, diabetic.Insulin, color="red", label="Diabetic", alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Insulin")
plt.legend()
plt.show()

# ### 3. **Feature Engineering:**
#

# ##### 3.1. Generating New Features

df.head()

# Describe the "Age" column to get statistical summary.
df["Age"].describe().T

# Create a new categorical feature based on age groups.
df["NEW_AGE_CAT"] = np.where(df["Age"] < 50, "mature", "senior")

# Categorize BMI values into different groups.
df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],
                       labels=["Underweight", "Healthy", "Overweight", "Obese"])

# Convert glucose values into categorical variables.
df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])

# Create a new categorical feature considering age and BMI together.
categories = ["underweightmature", "underweightsenior", "healthymature", "healthysenior", "overweightmature",
              "overweightsenior", "obesemature", "obesesenior"]
df["NEW_AGE_BMI_NOM"] = pd.Categorical(df["NEW_BMI"], categories=categories)

df.loc[(df["BMI"] < 18.5) & (df["Age"] < 50), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] < 50), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] < 50), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & (df["Age"] < 50), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"

df["NEW_INSULIN_BMI_RATIO"] = df["Insulin"] / df["BMI"]


# Generating Categorical Variables with Insulin
def set_insulin(dataframe):
    if 16 <= dataframe["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"


df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

df["NEW_GLUCOSE*INSULIN"] = df["Glucose"] * df["Insulin"]

df["NEW_GLUCOSE*PREGNANCIES"] = df["Glucose"] * df["Pregnancies"]

df.head()

df.columns = [col.upper() for col in df.columns]
df.head()
df.shape

# ### Step 3: Perform encoding

# Separating variables by their types
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Observations: 768
# Variables: 17
# cat_cols: 7
# num_cols: 10
# cat_but_car: 0
# num_but_cat: 3

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols
# ['NEW_AGE_CAT', 'NEW_INSULIN_SCORE']

# Label encoding for binary columns
for col in binary_cols:
    df = label_encoder(df, col)

df.head()

# One-Hot Encoding Process
# Updating the cat_cols list
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols


# ['NEW_AGE_BMI_NOM', 'NEW_AGE_GLUCOSE_NOM', 'NEW_BMI', 'NEW_GLUCOSE']

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)

df.shape
df.head()

# ### Step 4: Standardization for numerical variables.
#

num_cols
# ['PREGNANCIES',
#  'GLUCOSE',
#  'BLOODPRESSURE',
#  'SKINTHICKNESS',
#  'INSULIN',
#  'BMI',
#  'DIABETESPEDIGREEFUNCTION',
#  'AGE',
#  'NEW_GLUCOSE*INSULIN',
#  'NEW_GLUCOSE*PREGNANCIES']

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()

# ### 4. **Model Development:**
#

y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestClassifier(random_state=17)
rf_model.fit(X_train, y_train)

# ### 5. Model Evaluation

y_pred = rf_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Results
results = pd.DataFrame({"Models": ["Random Forest Classifier"],
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1 Score": f1,
                        "ROC AUC": roc_auc})

print(results)

# # Results and Conclusion

# # Correlation Analysis:
# The correlation between "Glucose" and "Outcome" is significantly high (0.493), indicating a strong relationship between blood glucose levels and diabetes.
# There's a notable correlation between "BMI" and "Outcome" (0.312), suggesting a link between body mass index and diabetes.
# A considerable correlation is observed between "Age" and "Outcome" (0.238), indicating an association between age and diabetes.
#
# # Model Performance:
# Upon reviewing the model outcomes:
#
# Models	Accuracy	Precision	Recall	F1 Score	ROC AUC
# Random Forest Classifier	0.779      0.795   0.544     0.646    0.731
#
# These metrics depict the predictive power of the model. The Random Forest Classifier achieves an accuracy of 77.9%, precision of 79.5%, recall of 54.4%, F1 score of 64.6%, and an ROC AUC score of 0.731.
#
# > # Conclusion:
# The strong correlations observed between "Glucose", "BMI", "Age", and "Outcome" indicate their significance in predicting diabetes risk.
# "Glucose" levels, "BMI", and "Age" appear to be crucial factors in predicting the likelihood of diabetes.
# Further feature engineering could be undertaken to enhance the model's performance. Additionally, experimenting with different models or ensemble techniques might improve predictions.
# Validation of the model's performance using additional test data would be prudent to ensure its robustness and generalizability.
# By leveraging these insights, healthcare practitioners can better assess diabetes risk factors and implement preventive measures effectively.
#
#
#