import os
import shutil

# 데이터 폴더 경로 설정
rgb_folder = "./rgb/bottle_4"  # RGB 이미지 폴더 경로
masks_folder = "./masks/bottle_4"  # 마스크 데이터 폴더 경로
# testset_folder = "./testset/bottle_4"  # 검증용 test 데이터 폴더 경로

# 새로운 저장 폴더 경로 설정
train_x_folder = "./train_x/bottle_4"  # train 폴더 경로
train_y_folder = "./train_y/bottle_4"  # validation 폴더 경로
# test_folder = "./test/bottle_4"  # test 폴더 경로

# 저장 폴더 생성
os.makedirs(train_x_folder, exist_ok=True)
os.makedirs(train_y_folder, exist_ok=True)
# os.makedirs(test_folder, exist_ok=True)

# RGB 이미지 (train) 처리 및 복사
rgb_files = os.listdir(rgb_folder)
for file in rgb_files:
    old_path = os.path.join(rgb_folder, file)
    if os.path.isfile(old_path):
        # "rgb_숫자_0"에서 숫자 추출
        number = file.split("_")[1]
        new_name = f"{int(number):07d}.png"  # 7자리 숫자로 변환
        new_path = os.path.join(train_x_folder, new_name)
        shutil.copy(old_path, new_path)
        print(f"Copied and renamed (train): {file} -> {new_name}")

# 마스크 데이터 (validation) 처리 및 복사
masks_files = os.listdir(masks_folder)
for file in masks_files:
    old_path = os.path.join(masks_folder, file)
    if os.path.isfile(old_path):
        # "01_숫자"에서 숫자 추출
        number = file.split("_")[1].split(".")[0]
        new_name = f"{int(number):07d}.png"  # 7자리 숫자로 변환
        new_path = os.path.join(train_y_folder, new_name)
        shutil.copy(old_path, new_path)
        print(f"Copied and renamed (validation): {file} -> {new_name}")

# # 검증용 test 데이터 처리 및 복사
# testset_files = os.listdir(testset_folder)
# for file in testset_files:
#     old_path = os.path.join(testset_folder, file)
#     if os.path.isfile(old_path):
#         # "rgb_숫자_0"에서 숫자 추출
#         number = file.split("_")[1]
#         new_name = f"{int(number):07d}.png"  # 7자리 숫자로 변환
#         new_path = os.path.join(test_folder, new_name)
#         shutil.copy(old_path, new_path)
#         print(f"Copied and renamed (test): {file} -> {new_name}")
