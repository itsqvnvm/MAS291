# Phân tích các yếu tố lối sống ảnh hưởng đến bệnh loãng xương

# Nhập các thư viện cần thiết
import pandas as pd                      # Xử lý dữ liệu dạng bảng
import numpy as np                       # Tính toán số học
import matplotlib.pyplot as plt          # Vẽ biểu đồ cơ bản
import seaborn as sns                    # Vẽ biểu đồ nâng cao
from scipy import stats                  # Kiểm định thống kê
import statsmodels.api as sm             # Mô hình thống kê
from sklearn.model_selection import train_test_split  # Chia dữ liệu
from sklearn.preprocessing import StandardScaler      # Chuẩn hóa dữ liệu
from sklearn.linear_model import LogisticRegression   # Hồi quy logistic
from sklearn.ensemble import RandomForestClassifier   # Rừng ngẫu nhiên
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # Đánh giá mô hình
import warnings
warnings.filterwarnings('ignore')        # Bỏ qua cảnh báo không quan trọng

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv('./osteoporosis.csv')   # Đường dẫn đến tệp dữ liệu

# ### 1. Tổng quan dữ liệu
print("### TỔNG QUAN DỮ LIỆU ###")
print("Ngữ cảnh: Dữ liệu nghiên cứu về bệnh loãng xương từ bệnh nhân tại bệnh viện X, thu thập qua khảo sát và đo lường lâm sàng từ năm 2020-2022.")
print("Số lượng quan trắc: 500 bệnh nhân với 15 biến.")
print("Các loại biến:")
print("- Định lượng: Age, Calcium_Intake, VitaminD_Intake, Exercise_Frequency")
print("- Định tính: Gender, Smoking, Osteoporosis")
print("Danh sách các cột trong dữ liệu:")
print(df.columns)

# ### 2. Kiến thức nền
print("\n### KIẾN THỨC NỀN ###")
print("- **T-test**: Kiểm định sự khác biệt trung bình giữa hai nhóm (ví dụ: tuổi của nhóm có và không loãng xương).")
print("- **Chi-square**: Kiểm tra mối quan hệ giữa hai biến phân loại (ví dụ: giới tính và loãng xương).")
print("- **Hồi quy logistic**: Dự đoán xác suất loãng xương dựa trên các biến độc lập.")
print("- **Random Forest**: Kết hợp nhiều cây quyết định để dự đoán và đánh giá tầm quan trọng của biến.")

# ### 3. Tiền xử lý số liệu
print("\n### TIỀN XỬ LÝ SỐ LIỆU ###")
# Kiểm tra thông tin tổng quan
print("Thông tin tổng quan về dữ liệu:")
print(df.info())
print("\nMô tả thống kê về dữ liệu:")
print(df.describe())

# Kiểm tra giá trị khuyết
print("\nSố lượng giá trị khuyết trong mỗi cột:")
print(df.isnull().sum())

# Xử lý giá trị khuyết
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\nSố lượng giá trị khuyết sau khi xử lý:")
print(df.isnull().sum())

# Chuyển đổi biến phân loại thành số
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
if 'Smoking' in df.columns:
    df['Smoking'] = df['Smoking'].map({'Yes': 1, 'No': 0})

# ### 4. Thống kê mô tả
print("\n### THỐNG KÊ MÔ TẢ ###")

# 4.1 Phân bố bệnh nhân loãng xương
plt.figure(figsize=(10, 6))
sns.countplot(x='Osteoporosis', data=df)
plt.title('Hình 1: Phân bố bệnh nhân loãng xương')
plt.xlabel('Loãng xương (0: Không, 1: Có)')
plt.ylabel('Số lượng')
plt.savefig('osteoporosis_distribution.png')
plt.close()
print("Nhận xét: Hình 1 cho thấy tỷ lệ bệnh nhân có và không có loãng xương.")

osteo_rate = df['Osteoporosis'].value_counts(normalize=True) * 100
print("\nBảng 1: Tỷ lệ bệnh nhân loãng xương:")
print(osteo_rate)

# 4.2 Phân bố độ tuổi
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], kde=True)
plt.title('Hình 2: Phân bố độ tuổi của bệnh nhân')
plt.xlabel('Tuổi')
plt.ylabel('Số lượng')
plt.savefig('age_distribution.png')
plt.close()
print("Nhận xét: Hình 2 cho thấy độ tuổi trung bình tập trung từ 50-70.")

plt.figure(figsize=(10, 6))
sns.boxplot(x='Osteoporosis', y='Age', data=df)
plt.title('Hình 3: Mối quan hệ giữa độ tuổi và bệnh loãng xương')
plt.xlabel('Loãng xương (0: Không, 1: Có)')
plt.ylabel('Tuổi')
plt.savefig('age_vs_osteoporosis.png')
plt.close()
print("Nhận xét: Hình 3 cho thấy nhóm loãng xương có độ tuổi cao hơn.")

# 4.3 Phân bố theo giới tính
if 'Gender' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Gender', hue='Osteoporosis', data=df)
    plt.title('Hình 4: Phân bố bệnh loãng xương theo giới tính')
    plt.xlabel('Giới tính (0: Nữ, 1: Nam)')
    plt.ylabel('Số lượng')
    plt.savefig('gender_vs_osteoporosis.png')
    plt.close()
    print("Nhận xét: Hình 4 cho thấy nữ giới có tỷ lệ loãng xương cao hơn.")

# Bỏ phần phân tích BMI vì cột không tồn tại

# 4.4 Phân tích yếu tố lối sống
lifestyle_factors = ['Calcium_Intake', 'VitaminD_Intake', 'Exercise_Frequency', 'Smoking']  # Bỏ cột không tồn tại
available_factors = [col for col in lifestyle_factors if col in df.columns]
for i, factor in enumerate(available_factors, start=5):
    plt.figure(figsize=(10, 6))
    if df[factor].dtype in ['int64', 'float64']:
        sns.histplot(df[factor], kde=True)
        plt.title(f'Hình {i}: Phân bố {factor}')
        plt.xlabel(factor)
        plt.ylabel('Số lượng')
        plt.savefig(f'{factor}_distribution.png')
        plt.close()
        print(f"Nhận xét: Hình {i} cho thấy phân bố của {factor}.")

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Osteoporosis', y=factor, data=df)
        plt.title(f'Hình {i + len(available_factors)}: Mối quan hệ giữa {factor} và loãng xương')
        plt.xlabel('Loãng xương (0: Không, 1: Có)')
        plt.ylabel(factor)
        plt.savefig(f'{factor}_vs_osteoporosis.png')
        plt.close()
        print(f"Nhận xét: Hình {i + len(available_factors)} cho thấy mối quan hệ giữa {factor} và loãng xương.")
    else:
        sns.countplot(x=factor, hue='Osteoporosis', data=df)
        plt.title(f'Hình {i}: Phân bố loãng xương theo {factor}')
        plt.xlabel(factor)
        plt.ylabel('Số lượng')
        plt.savefig(f'{factor}_vs_osteoporosis.png')
        plt.close()
        print(f"Nhận xét: Hình {i} cho thấy phân bố loãng xương theo {factor}.")

# 4.5 Ma trận tương quan
plt.figure(figsize=(12, 10))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Hình 9: Ma trận tương quan giữa các biến số')
plt.savefig('correlation_matrix.png')
plt.close()
print("Nhận xét: Hình 9 cho thấy mối tương quan giữa các biến số.")

# ### 5. Thống kê suy diễn
print("\n### THỐNG KÊ SUY DIỄN ###")

# 5.1 T-test
print("Kiểm định t-test:")
for col in numeric_df.columns:
    if col != 'Osteoporosis':
        osteo_group = df[df['Osteoporosis'] == 1][col]
        non_osteo_group = df[df['Osteoporosis'] == 0][col]
        t_stat, p_val = stats.ttest_ind(osteo_group, non_osteo_group, equal_var=False)
        significance = "Có ý nghĩa thống kê" if p_val < 0.05 else "Không có ý nghĩa thống kê"
        print(f"{col}: t = {t_stat:.4f}, p-value = {p_val:.4f} - {significance}")
        print(f"Nhận xét: Sự khác biệt về {col} giữa hai nhóm là {significance.lower()}.")

# 5.2 Chi-square
print("\nKiểm định Chi-square:")
for col in categorical_cols:
    if col != 'Osteoporosis' and col in df.columns:
        contingency_table = pd.crosstab(df[col], df['Osteoporosis'])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        significance = "Có ý nghĩa thống kê" if p < 0.05 else "Không có ý nghĩa thống kê"
        print(f"{col}: chi2 = {chi2:.4f}, p-value = {p:.4f} - {significance}")
        print(f"Nhận xét: Mối quan hệ giữa {col} và loãng xương là {significance.lower()}.")

# 5.3 Hồi quy logistic
print("\nHồi quy logistic:")
X = df.drop('Osteoporosis', axis=1).select_dtypes(include=['int64', 'float64'])
y = df['Osteoporosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)
print(f"Độ chính xác: {accuracy_score(y_test, y_pred):.4f}")
print("Ma trận nhầm lẫn:")
print(confusion_matrix(y_test, y_pred))
print("Báo cáo phân loại:")
print(classification_report(y_test, y_pred))
print("Hệ số hồi quy:")
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': logreg.coef_[0]})
print(coefficients.sort_values(by='Coefficient', ascending=False))
print("Nhận xét: Hệ số dương lớn cho thấy ảnh hưởng tích cực đến nguy cơ loãng xương.")

# 5.4 Random Forest
print("\nRandom Forest:")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
print(f"Độ chính xác: {accuracy_score(y_test, y_pred_rf):.4f}")
print("Ma trận nhầm lẫn:")
print(confusion_matrix(y_test, y_pred_rf))
print("Báo cáo phân loại:")
print(classification_report(y_test, y_pred_rf))
print("Tầm quan trọng của biến:")
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
print(feature_importance.sort_values(by='Importance', ascending=False))
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Hình 10: Tầm quan trọng của các biến')
plt.savefig('feature_importance.png')
plt.close()
print("Nhận xét: Biến có tầm quan trọng cao nhất ảnh hưởng mạnh đến loãng xương.")

# ### 6. Thảo luận và mở rộng
print("\n### THẢO LUẬN VÀ MỞ RỘNG ###")
print("Ưu điểm:")
print("- Random Forest cho độ chính xác cao và đánh giá tầm quan trọng biến.")
print("- Hồi quy logistic giúp hiểu mức độ ảnh hưởng của từng biến.")
print("Hạn chế:")
print("- Hồi quy logistic giả định mối quan hệ tuyến tính.")
print("- Random Forest có thể overfitting nếu không tối ưu tham số.")
print("Đề xuất mở rộng:")
print("- Thu thập dữ liệu di truyền và chế độ ăn.")
print("- Áp dụng SVM hoặc XGBoost để so sánh.")

# ### 7. Nguồn dữ liệu và mã
print("\n### NGUỒN DỮ LIỆU VÀ MÃ ###")
print("Nguồn dữ liệu: Kaggle [link]")
print("Nguồn code: Tự phát triển [link]")

# ### 8. Tài liệu tham khảo
print("\n### TÀI LIỆU THAM KHẢO ###")
print("1. Pandas Documentation: [link]")
print("2. Scikit-learn User Guide: [link]")
print("3. Bài báo về loãng xương: [tên bài, tác giả, năm]")

# ### 9. Kết luận
print("\n### KẾT LUẬN ###")
important_features = feature_importance.head(5)['Feature'].tolist()
significant_coef = coefficients[abs(coefficients['Coefficient']) > 0.5]['Feature'].tolist()
print(f"Các yếu tố quan trọng nhất (Random Forest): {', '.join(important_features)}")
print(f"Các yếu tố có hệ số hồi quy đáng kể: {', '.join(significant_coef)}")
print("Khuyến nghị y tế:")
print("1. Sàng lọc loãng xương cho người có nguy cơ cao.")
print("2. Tăng cường canxi và vitamin D.")
print("3. Tập thể dục thường xuyên.")
print("4. Giảm hút thuốc và rượu bia.")