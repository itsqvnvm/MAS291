import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.ticker as ticker

# 1. Đọc dữ liệu
df = pd.read_csv('./osteoporosis.csv')

# 2. Làm sạch dữ liệu
# Kiểm tra dữ liệu khuyết
print("Kiểm tra dữ liệu khuyết:")
print(df.isnull().sum())

# Xử lý dữ liệu khuyết (nếu cần)
# Với dữ liệu số, điền bằng giá trị trung bình
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

# Với dữ liệu phân loại, điền bằng giá trị phổ biến nhất
for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Kiểm tra lại sau khi xử lý
print("\nKiểm tra lại sau khi xử lý dữ liệu khuyết:")
print(df.isnull().sum().sum())

# 3. Làm rõ dữ liệu (Data visualization)
# Thống kê mô tả
print("\nThống kê mô tả:")
print(df.describe())

# Kiểm tra phân phối của các biến
print("\nThông tin dữ liệu:")
print(df.info())

# Thiết lập kích thước biểu đồ
plt.figure(figsize=(15, 10))

# Biểu đồ phân bố theo độ tuổi
plt.subplot(2, 3, 1)
sns.histplot(df['Age'], kde=True)
plt.title('Phân bố độ tuổi')

# Phân bố theo giới tính nếu có trường này
if 'Gender' in df.columns:
    plt.subplot(2, 3, 2)
    sns.countplot(x='Gender', data=df)
    plt.title('Phân bố theo giới tính')

# Phân bố loãng xương nếu có trường này
if 'Osteoporosis' in df.columns:
    plt.subplot(2, 3, 3)
    sns.countplot(x='Osteoporosis', data=df)
    plt.title('Phân bố bệnh loãng xương')

# Mối quan hệ giữa tuổi và BMD nếu có trường này
if 'BMD_value' in df.columns:
    plt.subplot(2, 3, 4)
    sns.scatterplot(x='Age', y='BMD_value', hue='Osteoporosis' if 'Osteoporosis' in df.columns else None, data=df)
    plt.title('Mối quan hệ giữa tuổi và mật độ xương')

# Mối quan hệ giữa BMI và BMD nếu có các trường này
if 'BMI' in df.columns and 'BMD_value' in df.columns:
    plt.subplot(2, 3, 5)
    sns.scatterplot(x='BMI', y='BMD_value', hue='Osteoporosis' if 'Osteoporosis' in df.columns else None, data=df)
    plt.title('Mối quan hệ giữa BMI và mật độ xương')

plt.tight_layout()
plt.savefig('osteoporosis_overview.png')
plt.show()

# Phân tích chi tiết hơn
plt.figure(figsize=(15, 10))

# Biểu đồ tương quan (heatmap)
plt.subplot(2, 2, 1)
# Chỉ lấy các cột số
numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation = numeric_df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Tương quan giữa các biến số')

# Boxplot cho các biến quan trọng theo tình trạng loãng xương
if 'Osteoporosis' in df.columns:
    # BMD theo tình trạng loãng xương nếu có trường BMD
    if 'BMD_value' in df.columns:
        plt.subplot(2, 2, 2)
        sns.boxplot(x='Osteoporosis', y='BMD_value', data=df)
        plt.title('BMD theo tình trạng loãng xương')
    
    # Calcium Intake theo tình trạng loãng xương nếu có trường này
    if 'Calcium_Intake' in df.columns:
        plt.subplot(2, 2, 3)
        sns.boxplot(x='Osteoporosis', y='Calcium_Intake', data=df)
        plt.title('Lượng Calcium theo tình trạng loãng xương')
    
    # Physical Activity theo tình trạng loãng xương nếu có trường này
    if 'Physical_Activity' in df.columns:
        plt.subplot(2, 2, 4)
        sns.boxplot(x='Osteoporosis', y='Physical_Activity', data=df)
        plt.title('Hoạt động thể chất theo tình trạng loãng xương')

plt.tight_layout()
plt.savefig('osteoporosis_detailed.png')
plt.show()

# Phân tích theo độ tuổi và giới tính
if 'Gender' in df.columns and 'Osteoporosis' in df.columns:
    plt.figure(figsize=(12, 6))
    
    # Tạo nhóm tuổi mới từ cột Age
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90+']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    # Tỷ lệ loãng xương theo độ tuổi và giới tính
    sns.catplot(x='Age_Group', y='Osteoporosis', hue='Gender', kind='point', data=df)
    plt.title('Tỷ lệ loãng xương theo độ tuổi và giới tính')
    plt.tight_layout()
    plt.savefig('osteoporosis_age_gender.png')
    plt.show()

# 4. Phân tích thống kê suy luận
print("\n=== Phân tích thống kê suy luận ===")

# Chuẩn bị dữ liệu cho mô hình dự đoán (nếu có cột Osteoporosis)
if 'Osteoporosis' in df.columns:
    # Chọn các biến đầu vào tiềm năng (điều chỉnh theo bộ dữ liệu thực tế)
    potential_features = [col for col in numeric_df.columns if col != 'Osteoporosis']
    X = df[potential_features]
    y = df['Osteoporosis']
    
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Xây dựng mô hình hồi quy logistic
    print("\nKết quả mô hình Hồi quy Logistic:")
    logistic_model = LogisticRegression(random_state=42)
    logistic_model.fit(X_train_scaled, y_train)
    
    # Dự đoán và đánh giá
    y_pred_logistic = logistic_model.predict(X_test_scaled)
    print("\nĐộ chính xác:", accuracy_score(y_test, y_pred_logistic))
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test, y_pred_logistic))
    print("\nMa trận nhầm lẫn:")
    print(confusion_matrix(y_test, y_pred_logistic))
    
    # Hệ số quan trọng
    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': logistic_model.coef_[0]
    })
    coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
    print("\nHệ số quan trọng trong mô hình hồi quy logistic:")
    print(coefficients)
    
    # Xây dựng mô hình Random Forest
    print("\nKết quả mô hình Random Forest:")
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Dự đoán và đánh giá
    y_pred_rf = rf_model.predict(X_test)
    print("\nĐộ chính xác:", accuracy_score(y_test, y_pred_rf))
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test, y_pred_rf))
    
    # Độ quan trọng của biến
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    print("\nĐộ quan trọng của biến trong mô hình Random Forest:")
    print(feature_importance)
    
    # Biểu đồ độ quan trọng của biến
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Độ quan trọng của các biến trong mô hình Random Forest')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
else:
    print("Không tìm thấy cột Osteoporosis trong dữ liệu để thực hiện phân tích dự đoán.")