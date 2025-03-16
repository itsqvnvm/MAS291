import csv
from collections import defaultdict

data = []

with open("./osteoporosis.csv", mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)
    header = next(reader)
    for row in reader:
        data.append(row)


#---------------------------------------+
# Định lý Bayes & Xác suất có điều kiện +
#-------------------------------------- +

sample_size = len(data)

age_data = []
gender_data = []
hormonal_changes_data = []
family_history_data = []
race_data = []
body_weight_data = []
calcium_intake_data = []
vitamin_d_intake_data = []
physical_activity_data = []
smoking_data = []
alcohol_consumption_data = []
medical_conditions_data = []
medications_data = []
prior_fractures_data = []

num_osteoporosis = 0 # Số người bị loãng xương

max_age = 101
num_of_times_age = [0] * max_age # Số lần xuất hiện độ tuổi đó trong tập dữ liệu
num_of_times_gender = {
    'Male': 0,
    'Female': 0
}
num_of_times_hormonal = {
    'Normal': 0,
    'Postmenopausal': 0
}
num_of_times_family_history = {
    'Yes': 0,
    'No': 0
}
num_of_times_race = defaultdict(int) # Từ điển tự động thêm nếu key chưa tồn tại
num_of_times_body_weight = defaultdict(int)
num_of_times_calcium_intake = defaultdict(int)
num_of_times_vitamin_d_intake = defaultdict(int)
num_of_times_physical_activity = defaultdict(int)
num_of_times_smoking = defaultdict(int)
num_of_times_alcohol_consumption = defaultdict(int)
num_of_times_medical_conditions = defaultdict(int)
num_of_times_medications = defaultdict(int)
num_of_times_prior_fractures = defaultdict(int)

for i in data: # Làm sạch dữ liệu và chuẩn hóa
    yes_or_no = int(i[15]) # 1 là có bệnh 0 là không bệnh
    num_osteoporosis+=yes_or_no
    age = int(i[1])
    num_of_times_age[age]+=1
    age_data.append((age, yes_or_no))
    gender_data.append((i[2], yes_or_no))
    num_of_times_gender[i[2]]+=1
    hormonal_changes_data.append((i[3], yes_or_no))
    num_of_times_hormonal[i[3]]+=1
    family_history_data.append((i[4], yes_or_no))
    num_of_times_family_history[i[4]]+=1
    race_data.append((i[5], yes_or_no))
    num_of_times_race[i[5]]+=1
    body_weight_data.append((i[6], yes_or_no))
    num_of_times_body_weight[i[6]]+=1
    calcium_intake_data.append((i[7], yes_or_no))
    num_of_times_calcium_intake[i[7]]+=1
    vitamin_d_intake_data.append((i[8], yes_or_no))
    num_of_times_vitamin_d_intake[i[8]]+=1
    physical_activity_data.append((i[9], yes_or_no))
    num_of_times_physical_activity[i[9]]+=1
    smoking_data.append((i[10], yes_or_no))
    num_of_times_smoking[i[10]]+=1
    alcohol_consumption_data.append((i[11], yes_or_no))
    num_of_times_alcohol_consumption[i[11]]+=1
    medical_conditions_data.append((i[12], yes_or_no))
    num_of_times_medical_conditions[i[12]]+=1
    medications_data.append((i[13], yes_or_no))
    num_of_times_medications[i[13]]+=1
    prior_fractures_data.append((i[14], yes_or_no))
    num_of_times_prior_fractures[i[14]]+=1

P_O = num_osteoporosis / sample_size

# Xác suất bị loãng xương dựa trên đặc điểm độ tuổi P(O|A) = P(A|0) * P(O) / P(A)
P_A = [0] * max_age # Xác suất xuất hiện độ tuổi đó trong tập dữ liệu
P_A_O = [0] * max_age # Xác suất người có độ tuổi đó mắc bệnh
for i in range(1, max_age):
    if num_of_times_age[i] > 0:
        P_A[i] = num_of_times_age[i] / sample_size
        dem = 0
        for j in age_data:
            if j[0] == i and j[1] == 1:
                dem+=1
        P_A_O[i] = dem / num_osteoporosis
P_O_A = [0] * max_age # Xác suất bị loãng xương dựa trên độ tuổi
for i in range(1, max_age):
    if P_A[i] > 0:
        P_O_A[i] = P_A_O[i] * P_O / P_A[i]


# Xác suất người bị loãng xương dựa trên giới tính P(O|G) = P(G|O) * P(O) / P(G)
P_G = {
    'Male': num_of_times_gender['Male'] / sample_size,
    'Female': num_of_times_gender['Female'] / sample_size
} # Xác suất xuất hiện giới tính đó trong tập dữ liệu

P_G_O = {
    'Male': 0,
    'Female': 0
} # Xác suất người có giới tính đó mắc bệnh
count_m = 0
count_f = 0
for i in gender_data:
    if i[0] == 'Male' and i[1] == 1:
        count_m+=1
    elif i[0] == 'Female' and i[1] == 1:
        count_f+=1
P_G_O['Male'] = count_m / num_osteoporosis
P_G_O['Female'] = count_f / num_osteoporosis

P_O_G = {
    'Male': 0,
    'Female': 0
}
P_O_G['Male'] = P_G_O['Male'] * P_O / P_G['Male']
P_O_G['Female'] = P_G_O['Female'] * P_O / P_G['Female']


# Xác suất người bị loãng xương dựa trên hooc-môn P(O|HC) = P(HC|O) * P(O) / P(HC)
P_HC = {
    'Normal': num_of_times_hormonal['Normal'] / sample_size,
    'Postmenopausal': num_of_times_hormonal['Postmenopausal'] / sample_size
} # Xác suất xuất hiện ghi nhận sự thay đổi hooc-môn đó trong tập dữ liệu
P_HC_O = {
    'Normal': 0,
    'Postmenopausal': 0
} # Xác suất người có sự thay đổi hooc-môn đó mắc bệnh
count_n = 0
count_p = 0
for i in hormonal_changes_data:
    if i[0] == 'Normal' and i[1] == 1:
        count_n+=1
    elif i[0] == 'Postmenopausal' and i[1] == 1:
        count_p+=1
P_HC_O['Normal'] = count_n / num_osteoporosis
P_HC_O['Postmenopausal'] = count_p / num_osteoporosis

P_O_HC = {
    'Normal': 0,
    'Postmenopausal': 0
}
P_O_HC['Normal'] = P_HC_O['Normal'] * P_O / P_HC['Normal']
P_O_HC['Postmenopausal'] = P_HC_O['Postmenopausal'] * P_O / P_HC['Postmenopausal']


# Xác suất người bị loãng xương dựa trên tiền sử của gia đình P(O|FH) = P(FH|O) * P(O) / P(FH)
P_FH = {
    'Yes': num_of_times_family_history['Yes'] / sample_size,
    'No': num_of_times_family_history['No'] / sample_size
} # Xác suất xuất hiện có hoặc không ghi nhận tiền sử gia đình trong tập dữ liệu
P_FH_O = {
    'Yes': 0,
    'No': 0
} # Xác suất người có hoặc không có tiền sử gia đình mắc bệnh
count_y = 0
count_n = 0
for i in family_history_data:
    if i[0] == 'Yes' and i[1] == 1:
        count_y+=1
    elif i[0] == 'No' and i[1] == 1:
        count_n+=1
P_FH_O['Yes'] = count_y / num_osteoporosis
P_FH_O['No'] = count_n / num_osteoporosis

P_O_FH = {
    'Yes': 0,
    'No': 0
}
P_O_FH['Yes'] = P_FH_O['Yes'] * P_O / P_FH['Yes']
P_O_FH['No'] = P_FH_O['No'] * P_O / P_FH['No']


# Xác suất người bị loãng xương dựa trên chủng tộc P(O|R) = P(R|O) * P(O) / P(R)
P_R = dict.fromkeys(num_of_times_race.keys(), 0)
for key, value in num_of_times_race.items():
    P_R[key] = value / sample_size # Xác suất xuất hiện chủng tộc trong tập dữ liệu

P_R_O = dict.fromkeys(num_of_times_race.keys(), 0) # Xác suất chủng tộc mắc bệnh loãng xương
for i in race_data:
    if i[1] == 1:
        P_R_O[i[0]]+=1

for key, value in P_R_O.items():
    P_R_O[key] = value / num_osteoporosis

P_O_R = dict.fromkeys(num_of_times_race.keys(), 0)
for key, value in P_O_R.items():
    P_O_R[key] = P_R_O[key] * P_O / P_R[key]


# Xác suất người bị loãng xương dựa trên cân nặng cơ thể P(O|BW) = P(BW|O) * P(O) / P(BW)
P_BW = dict.fromkeys(num_of_times_body_weight.keys(), 0)
for key, value in num_of_times_body_weight.items():
    P_BW[key] = value / sample_size # Xác suất xuất hiện kiểu cân nặng cơ thể trong tập dữ liệu

P_BW_O = dict.fromkeys(num_of_times_body_weight.keys(), 0) # Xác suất kiểu cân nặng mắc bệnh loãng xương
for i in body_weight_data:
    if i[1] == 1:
        P_BW_O[i[0]]+=1

for key, value in P_BW_O.items():
    P_BW_O[key] = value / num_osteoporosis

P_O_BW = dict.fromkeys(num_of_times_body_weight.keys(), 0)
for key, value in P_O_BW.items():
    P_O_BW[key] = P_BW_O[key] * P_O / P_BW[key]

# Xác suất người bị loãng xương dựa trên lượng canxi hấp thụ P(O|CI) = P(CI|O) * P(O) / P(CI)
P_CI = dict.fromkeys(num_of_times_calcium_intake.keys(), 0)
for key, value in num_of_times_calcium_intake.items():
    P_CI[key] = value / sample_size # Xác suất xuất hiện các mức độ hấp thụ canxi trong tập dữ liệu

P_CI_O = dict.fromkeys(num_of_times_calcium_intake.keys(), 0) # Xác suất mức độ hấp thụ canxi sẽ gây bệnh loãng xương
for i in calcium_intake_data:
    if i[1] == 1:
        P_CI_O[i[0]]+=1

for key, value in P_CI_O.items():
    P_CI_O[key] = value / num_osteoporosis

P_O_CI = dict.fromkeys(num_of_times_calcium_intake.keys(), 0)
for key, value in P_O_CI.items():
    P_O_CI[key] = P_CI_O[key] * P_O / P_CI[key]


# Xác suất người bị loãng xương dựa trên lượng hấp thụ vitamin D P(O|VDI) = P(VDI|O) * P(O) / P(VDI)
P_VDI = dict.fromkeys(num_of_times_vitamin_d_intake.keys(), 0)
for key, value in num_of_times_vitamin_d_intake.items():
    P_VDI[key] = value / sample_size # Xác suất xuất hiện các lượng hấp thụ vitamin D trong tập dữ liệu

P_VDI_O = dict.fromkeys(num_of_times_vitamin_d_intake.keys(), 0) # Xác suất lượng hấp thụ vitamin D sẽ gây bệnh loãng xương
for i in vitamin_d_intake_data:
    if i[1] == 1:
        P_VDI_O[i[0]]+=1

for key, value in P_VDI_O.items():
    P_VDI_O[key] = value / num_osteoporosis

P_O_VDI = dict.fromkeys(num_of_times_vitamin_d_intake.keys(), 0)
for key, value in P_O_VDI.items():
    P_O_VDI[key] = P_VDI_O[key] * P_O / P_VDI[key]

# Xác suất người bị loãng xương dựa trên vận động thể chất P(O|PA) = P(PA|O) * P(O) / P(PA)
P_PA = dict.fromkeys(num_of_times_physical_activity.keys(), 0)
for key, value in num_of_times_physical_activity.items():
    P_PA[key] = value / sample_size # Xác suất xuất hiện các mức độ vận động thể chất trong tập dữ liệu

P_PA_O = dict.fromkeys(num_of_times_physical_activity.keys(), 0) # Xác suất mức độ vận động thể chất của người bệnh loãng xương
for i in physical_activity_data:
    if i[1] == 1:
        P_PA_O[i[0]]+=1

for key, value in P_PA_O.items():
    P_PA_O[key] = value / num_osteoporosis

P_O_PA = dict.fromkeys(num_of_times_physical_activity.keys(), 0)
for key, value in P_O_PA.items():
    P_O_PA[key] = P_PA_O[key] * P_O / P_PA[key]

# Xác suất người bị loãng xương dựa trên việc có hút thuốc hay không P(O|S) = P(S|O) * P(O) / P(S)
P_S = dict.fromkeys(num_of_times_smoking.keys(), 0)
for key, value in num_of_times_smoking.items():
    P_S[key] = value / sample_size # Xác suất xuất hiện có hoặc không hút thuốc trong tập dữ liệu

P_S_O = dict.fromkeys(num_of_times_smoking.keys(), 0) # Xác suất xuất hiện có hoặc không hút thuốc của người bệnh loãng xương
for i in smoking_data:
    if i[1] == 1:
        P_S_O[i[0]]+=1

for key, value in P_S_O.items():
    P_S_O[key] = value / num_osteoporosis

P_O_S = dict.fromkeys(num_of_times_smoking.keys(), 0)
for key, value in P_O_S.items():
    P_O_S[key] = P_S_O[key] * P_O / P_S[key]

# Xác suất người bị loãng xương dựa trên tiêu thụ rượu P(O|AC) = P(AC|O) * P(O) / P(AC)
P_AC = dict.fromkeys(num_of_times_alcohol_consumption.keys(), 0)
for key, value in num_of_times_alcohol_consumption.items():
    P_AC[key] = value / sample_size # Xác suất xuất hiện các mức độ tiêu thụ rượu trong tập dữ liệu

P_AC_O = dict.fromkeys(num_of_times_alcohol_consumption.keys(), 0) # Xác suất xuất hiện của các mức độ tiêu thụ rượu người bệnh loãng xương
for i in alcohol_consumption_data:
    if i[1] == 1:
        P_AC_O[i[0]]+=1

for key, value in P_AC_O.items():
    P_AC_O[key] = value / num_osteoporosis

P_O_AC = dict.fromkeys(num_of_times_alcohol_consumption.keys(), 0)
for key, value in P_O_AC.items():
    P_O_AC[key] = P_AC_O[key] * P_O / P_AC[key]

# Xác suất người bị loãng xương dựa trên tình trạng y tế của người đó P(O|MC) = P(MC|O) * P(O) / P(MC)
P_MC = dict.fromkeys(num_of_times_medical_conditions.keys(), 0)
for key, value in num_of_times_medical_conditions.items():
    P_MC[key] = value / sample_size # Xác suất xuất hiện các tình trạng y tế trong tập dữ liệu

P_MC_O = dict.fromkeys(num_of_times_medical_conditions.keys(), 0) # Xác suất xuất hiện các tình trạng y tế của người bệnh loãng xương
for i in medical_conditions_data:
    if i[1] == 1:
        P_MC_O[i[0]]+=1

for key, value in P_MC_O.items():
    P_MC_O[key] = value / num_osteoporosis

P_O_MC = dict.fromkeys(num_of_times_medical_conditions.keys(), 0)
for key, value in P_O_MC.items():
    P_O_MC[key] = P_MC_O[key] * P_O / P_MC[key]

# Xác suất người bị loãng xương dựa trên loại thuốc người đó sử dụng P(O|M) = P(M|O) * P(O) / P(M)
P_M = dict.fromkeys(num_of_times_medications.keys(), 0)
for key, value in num_of_times_medications.items():
    P_M[key] = value / sample_size # Xác suất xuất hiện loại thuốc trong tập dữ liệu

P_M_O = dict.fromkeys(num_of_times_medications.keys(), 0) # Xác suất xuất hiện các loại thuốc của người bệnh loãng xương
for i in medications_data:
    if i[1] == 1:
        P_M_O[i[0]]+=1

for key, value in P_M_O.items():
    P_M_O[key] = value / num_osteoporosis

P_O_M = dict.fromkeys(num_of_times_medications.keys(), 0)
for key, value in P_O_M.items():
    P_O_M[key] = P_M_O[key] * P_O / P_M[key]


# Xác suất người bị loãng xương dựa trên việc bị gãy xương trước đó P(O|PF) = P(PF|O) * P(O) / P(PF)
P_PF = dict.fromkeys(num_of_times_prior_fractures.keys(), 0)
for key, value in num_of_times_prior_fractures.items():
    P_PF[key] = value / sample_size # Xác suất xuất hiện có hay không gãy xương trước đó trong tập dữ liệu

P_PF_O = dict.fromkeys(num_of_times_prior_fractures.keys(), 0) # Xác suất xuất hiện việc có hay không gãy xương trước đó của người bệnh loãng xương
for i in prior_fractures_data:
    if i[1] == 1:
        P_PF_O[i[0]]+=1

for key, value in P_PF_O.items():
    P_PF_O[key] = value / num_osteoporosis

P_O_PF = dict.fromkeys(num_of_times_prior_fractures.keys(), 0)
for key, value in P_O_PF.items():
    P_O_PF[key] = P_PF_O[key] * P_O / P_PF[key]



# #Thực thi nhập dữ liệu từ user để đưa ra kết luận
# columns = [
#     "Age", "Gender", "Hormonal Changes", "Family History", "Race/Ethnicity",
#     "Body Weight", "Calcium Intake", "Vitamin D Intake", "Physical Activity",
#     "Smoking", "Alcohol Consumption", "Medical Conditions", "Medications",
#     "Prior Fractures"
# ]

# i = input("Nhập thông tin bên dưới đây và các đặc điểm phải ngăn cách nhau bằng dấu phẩy:\n").split(",")
# result = P_O
# result = result * P_O_A[int(i[0])] * P_O_G[i[1]] * P_O_HC[i[2]] * P_O_FH[i[3]] * P_O_R[i[4]] * P_O_BW[i[5]] * P_O_CI[i[6]]
# result = result * P_O_VDI[i[7]] * P_O_PA[i[8]] * P_O_S[i[9]] * P_O_AC[i[10]] * P_O_MC[i[11]] * P_O_M[i[12]] * P_O_PF[i[13]]
# divisor = P_A[int(i[0])] * P_G[i[1]] * P_HC[i[2]] * P_FH[i[3]] * P_R[i[4]] * P_BW[i[5]] * P_CI[i[6]] * P_VDI[i[7]]
# divisor = divisor * P_PA[i[8]] * P_S[i[9]] * P_AC[i[10]] * P_MC[i[11]] * P_M[i[12]] * P_PF[i[13]]
# result = result / divisor
# if result > 1: result = 1
# result = result * 100

# print(f"Nguy cơ mắc bệnh loãng xương của bạn là: {result:.2f}%")