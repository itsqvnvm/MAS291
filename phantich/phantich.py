# Phân tích các yếu tố lối sống ảnh hưởng đến bệnh loãng xương
# Bước 1: Đọc dữ liệu (Import data)

# Install required libraries
# pip3 install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Đọc dữ liệu
df = pd.read_csv('./osteoporosis.csv')

# Bước 2: Làm sạch dữ liệu (Data cleaning)
# Kiểm tra thông tin tổng quan
print("Thông tin tổng quan về dữ liệu:")
print(df.info())
print("\nMô tả thống kê về dữ liệu:")
print(df.describe())

# Kiểm tra giá trị khuyết (NA)
print("\nSố lượng giá trị khuyết trong mỗi cột:")
print(df.isnull().sum())

# Xử lý giá trị khuyết (NA)
# Đối với dữ liệu số, thay thế bằng giá trị trung bình
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

# Đối với dữ liệu phân loại, thay thế bằng giá trị phổ biến nhất
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Kiểm tra lại giá trị khuyết sau khi xử lý
print("\nSố lượng giá trị khuyết sau khi xử lý:")
print(df.isnull().sum())

# Bước 3: Làm rõ dữ liệu (Data visualization)
# Chuyển đổi biến (nếu cần thiết)
# Giả sử có biến phân loại cần chuyển đổi thành số
# Ví dụ: df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# Thống kê mô tả (Descriptive statistic)
# 3.1 Phân tích biến phụ thuộc (Osteoporosis)
plt.figure(figsize=(10, 6))
if 'Osteoporosis' in df.columns:
    sns.countplot(x='Osteoporosis', data=df)
    plt.title('Phân bố bệnh nhân loãng xương')
    plt.xlabel('Loãng xương (0: Không, 1: Có)')
    plt.ylabel('Số lượng')
    plt.savefig('osteoporosis_distribution.png')
    plt.close()

    # Tỷ lệ bệnh nhân loãng xương
    osteo_rate = df['Osteoporosis'].value_counts(normalize=True) * 100
    print("\nTỷ lệ bệnh nhân loãng xương:")
    print(osteo_rate)

# 3.2 Phân tích theo độ tuổi
plt.figure(figsize=(10, 6))
if 'Age' in df.columns:
    sns.histplot(df['Age'], kde=True)
    plt.title('Phân bố độ tuổi của bệnh nhân')
    plt.xlabel('Tuổi')
    plt.ylabel('Số lượng')
    plt.savefig('age_distribution.png')
    plt.close()

    # Phân tích loãng xương theo độ tuổi
    plt.figure(figsize=(10, 6))
    if 'Osteoporosis' in df.columns:
        sns.boxplot(x='Osteoporosis', y='Age', data=df)
        plt.title('Mối quan hệ giữa độ tuổi và bệnh loãng xương')
        plt.xlabel('Loãng xương (0: Không, 1: Có)')
        plt.ylabel('Tuổi')
        plt.savefig('age_vs_osteoporosis.png')
        plt.close()

# 3.3 Phân tích theo giới tính
plt.figure(figsize=(10, 6))
if 'Gender' in df.columns and 'Osteoporosis' in df.columns:
    gender_osteo = pd.crosstab(df['Gender'], df['Osteoporosis'])
    gender_osteo.plot(kind='bar', stacked=True)
    plt.title('Phân bố bệnh loãng xương theo giới tính')
    plt.xlabel('Giới tính')
    plt.ylabel('Số lượng')
    plt.xticks(rotation=0)
    plt.savefig('gender_vs_osteoporosis.png')
    plt.close()

# 3.4 Phân tích BMI
plt.figure(figsize=(10, 6))
if 'BMI' in df.columns:
    sns.histplot(df['BMI'], kde=True)
    plt.title('Phân bố chỉ số BMI')
    plt.xlabel('BMI')
    plt.ylabel('Số lượng')
    plt.savefig('bmi_distribution.png')
    plt.close()

    # Phân tích loãng xương theo BMI
    plt.figure(figsize=(10, 6))
    if 'Osteoporosis' in df.columns:
        sns.boxplot(x='Osteoporosis', y='BMI', data=df)
        plt.title('Mối quan hệ giữa BMI và bệnh loãng xương')
        plt.xlabel('Loãng xương (0: Không, 1: Có)')
        plt.ylabel('BMI')
        plt.savefig('bmi_vs_osteoporosis.png')
        plt.close()

# 3.5 Phân tích các yếu tố lối sống
lifestyle_factors = ['Calcium_Intake', 'VitaminD_Intake', 'Exercise_Frequency', 'Smoking', 'Alcohol_Consumption']
available_factors = [col for col in lifestyle_factors if col in df.columns]

for factor in available_factors:
    plt.figure(figsize=(10, 6))
    if df[factor].dtype in ['int64', 'float64']:
        sns.histplot(df[factor], kde=True)
        plt.title(f'Phân bố {factor}')
        plt.xlabel(factor)
        plt.ylabel('Số lượng')
        plt.savefig(f'{factor}_distribution.png')
        plt.close()

        # Mối quan hệ với loãng xương
        if 'Osteoporosis' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Osteoporosis', y=factor, data=df)
            plt.title(f'Mối quan hệ giữa {factor} và bệnh loãng xương')
            plt.xlabel('Loãng xương (0: Không, 1: Có)')
            plt.ylabel(factor)
            plt.savefig(f'{factor}_vs_osteoporosis.png')
            plt.close()
    else:
        if 'Osteoporosis' in df.columns:
            factor_osteo = pd.crosstab(df[factor], df['Osteoporosis'])
            factor_osteo.plot(kind='bar', stacked=True)
            plt.title(f'Phân bố bệnh loãng xương theo {factor}')
            plt.xlabel(factor)
            plt.ylabel('Số lượng')
            plt.xticks(rotation=45)
            plt.savefig(f'{factor}_vs_osteoporosis.png')
            plt.close()

# 3.6 Ma trận tương quan
plt.figure(figsize=(12, 10))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Ma trận tương quan giữa các biến số')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Bước 4: Phân tích thống kê (Inferential statistic)
print("\n--- PHÂN TÍCH THỐNG KÊ SUY LUẬN ---")

# 4.1 Kiểm định t-test đối với các biến số giữa nhóm có và không có loãng xương
if 'Osteoporosis' in df.columns:
    print("\nKiểm định t-test đối với các biến số:")
    for col in numeric_df.columns:
        if col != 'Osteoporosis' and 'Osteoporosis' in numeric_df.columns:
            osteo_group = df[df['Osteoporosis'] == 1][col]
            non_osteo_group = df[df['Osteoporosis'] == 0][col]
            t_stat, p_val = stats.ttest_ind(osteo_group, non_osteo_group, equal_var=False)
            significance = "Có ý nghĩa thống kê" if p_val < 0.05 else "Không có ý nghĩa thống kê"
            print(f"{col}: t = {t_stat:.4f}, p-value = {p_val:.4f} - {significance}")

# 4.2 Kiểm định Chi-square đối với các biến phân loại
if 'Osteoporosis' in df.columns:
    print("\nKiểm định Chi-square đối với các biến phân loại:")
    for col in categorical_cols:
        if col != 'Osteoporosis':
            contingency_table = pd.crosstab(df[col], df['Osteoporosis'])
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            significance = "Có ý nghĩa thống kê" if p < 0.05 else "Không có ý nghĩa thống kê"
            print(f"{col}: chi2 = {chi2:.4f}, p-value = {p:.4f} - {significance}")

# 4.3 Phân tích hồi quy logistic
if 'Osteoporosis' in df.columns:
    print("\nHồi quy logistic để dự đoán loãng xương:")
    # Chuẩn bị dữ liệu
    X = df.drop('Osteoporosis', axis=1)
    # Loại bỏ các biến phân loại
    X = X.select_dtypes(include=['int64', 'float64'])
    y = df['Osteoporosis']
    
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Xây dựng mô hình hồi quy logistic
    logreg = LogisticRegression(random_state=42)
    logreg.fit(X_train_scaled, y_train)
    
    # Dự đoán
    y_pred = logreg.predict(X_test_scaled)
    
    # Đánh giá mô hình
    print("\nĐánh giá mô hình hồi quy logistic:")
    print(f"Độ chính xác: {accuracy_score(y_test, y_pred):.4f}")
    print("\nMa trận nhầm lẫn:")
    print(confusion_matrix(y_test, y_pred))
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test, y_pred))
    
    # Hệ số hồi quy
    print("\nHệ số của các biến trong mô hình hồi quy logistic:")
    coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': logreg.coef_[0]})
    coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
    print(coefficients)

# 4.4 Random Forest để đánh giá tầm quan trọng của các biến
if 'Osteoporosis' in df.columns:
    print("\nRandom Forest để đánh giá tầm quan trọng của các biến:")
    # Xây dựng mô hình Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # Dự đoán
    y_pred_rf = rf.predict(X_test_scaled)
    
    # Đánh giá mô hình
    print("\nĐánh giá mô hình Random Forest:")
    print(f"Độ chính xác: {accuracy_score(y_test, y_pred_rf):.4f}")
    print("\nMa trận nhầm lẫn:")
    print(confusion_matrix(y_test, y_pred_rf))
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test, y_pred_rf))
    
    # Tầm quan trọng của các biến
    print("\nTầm quan trọng của các biến trong mô hình Random Forest:")
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    print(feature_importance)
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Tầm quan trọng của các biến trong dự đoán bệnh loãng xương')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# Kết luận
print("\n--- KẾT LUẬN ---")
print("Dựa trên phân tích thống kê, các yếu tố sau đây có mối liên quan đáng kể đến nguy cơ loãng xương:")

# Liệt kê các yếu tố quan trọng dựa trên phân tích thống kê
if 'Osteoporosis' in df.columns and 'feature_importance' in locals():
    important_features = feature_importance.head(5)['Feature'].tolist()
    print(f"- Các yếu tố quan trọng nhất (theo Random Forest): {', '.join(important_features)}")

if 'Osteoporosis' in df.columns and 'coefficients' in locals():
    significant_coef = coefficients[abs(coefficients['Coefficient']) > 0.5]['Feature'].tolist()
    print(f"- Các yếu tố có hệ số hồi quy đáng kể: {', '.join(significant_coef)}")

print("\nCác khuyến nghị y tế:")
print("1. Tăng cường sàng lọc loãng xương đối với những người có các yếu tố nguy cơ cao.")
print("2. Khuyến khích tăng cường chế độ ăn giàu canxi và vitamin D.")
print("3. Khuyến khích tập thể dục thường xuyên để tăng cường mật độ xương.")
print("4. Giảm hút thuốc và tiêu thụ rượu bia để giảm nguy cơ loãng xương.")