import pandas as pd

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv("./osteoporosis.csv") # Replace "your_data.csv" with the actual filename

print(df.head())  # Display the first few rows to inspect the data

# Check for missing values in each column
print(df.isnull().sum())

# Option 1: Remove rows with any missing values (be careful, this can lose data)
# df = df.dropna()

# Option 2: Impute missing values (replace them with a reasonable estimate)
# For numerical columns, you might use the mean or median
for col in df.select_dtypes(include=['number']).columns:
    df[col].fillna(df[col].mean(), inplace=True) # Replace NA with the mean

# For categorical columns, you might use the mode (most frequent value)
for col in df.select_dtypes(include=['object']).columns:  # 'object' is often used for strings
    df[col].fillna(df[col].mode()[0], inplace=True) # Replace NA with the most frequent category

print(df.isnull().sum()) # Verify that missing values have been handled

import matplotlib.pyplot as plt
import seaborn as sns  # For more advanced visualizations

# 3.1 Variable Transformation (if needed)
# Example: Create age groups
# df['age_group'] = pd.cut(df['Age'], bins=[0, 50, 60, 70, 80], labels=['<50', '50-60', '60-70', '70+'])

# 3.2 Descriptive Statistics
print(df.describe())  # Summary statistics for numerical columns

# Visualizations:
# Histograms for numerical variables
for col in df.select_dtypes(include=['number']).columns:
    plt.figure()
    sns.histplot(df[col], kde=True) #kde adds a kernel density estimate
    plt.title(f'Distribution of {col}')
    plt.show()

# Bar charts for categorical variables
for col in df.select_dtypes(include=['object']).columns:
    plt.figure()
    sns.countplot(data=df, x=col)
    plt.title(f'Count of {col}')
    plt.xticks(rotation=45) # Rotate labels if needed
    plt.show()

# # Correlation heatmap for numerical variables
# plt.figure()
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# plt.show()

# Correlation heatmap for numerical variables only
numeric_df = df.select_dtypes(include=['number'])
plt.figure()
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

from scipy import stats
import statsmodels.api as sm

# 4.1 Content of Method (Example: Logistic Regression)
# Assuming 'Osteoporosis' is your target variable (0 = no, 1 = yes)
X = df[['Age', 'Gender', 'BMI']]  # Example features
y = df['Osteoporosis']

# Add a constant to the model for the intercept
X = sm.add_constant(X)

# Fit the logistic regression model
model = sm.Logit(y, X).fit()

# 4.2 Results
print(model.summary()) # Provides detailed results of the model (coefficients, p-values, etc.)

# 4.3 Code to implement the above description and requirements:
# The code above already implements the steps described in 4.1 and 4.2.  You can further use this model for prediction on new data.
