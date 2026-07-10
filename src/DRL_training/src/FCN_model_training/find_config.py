import os
import numpy as np
from PIL import Image


def find_global_min_size(image_paths):
    """전체 데이터셋(훈련 데이터)에서 가장 작은 폭과 높이를 찾음."""
    min_width = float("inf")
    min_height = float("inf")

    for img_path in image_paths:  # 경로 리스트를 순회
        with Image.open(img_path) as img:
            width, height = img.size
            min_width = min(min_width, width)
            min_height = min(min_height, height)

    return (min_height, min_width)


def calculate_mean_std(image_paths, batch_size=1000):
    """정확한 전체 mean과 std를 구하기 위해 배치 단위로 계산"""
    total_pixels = 0
    mean_sum = np.zeros(3)
    var_sum = np.zeros(3)

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_pixels = 0
        batch_mean = np.zeros(3)
        batch_var = np.zeros(3)

        for img_path in batch_paths:
            with Image.open(img_path) as img:
                img = np.array(img).astype(np.float32) / 255.0
                if img.ndim == 2:  # Grayscale image
                    img = np.stack([img] * 3, axis=-1)
                elif img.shape[-1] == 4:  # RGBA image
                    img = img[..., :3]
                h, w, c = img.shape
                pixels = img.reshape(-1, 3)
                batch_pixels += len(pixels)
                batch_mean += pixels.sum(axis=0)
                batch_var += (pixels**2).sum(axis=0)

        batch_mean /= batch_pixels
        batch_var = (batch_var / batch_pixels) - (batch_mean**2)

        mean_sum += batch_pixels * batch_mean
        var_sum += batch_pixels * (batch_var + batch_mean**2)
        total_pixels += batch_pixels

    mean = mean_sum / total_pixels
    variance = (var_sum / total_pixels) - (mean**2)
    std = np.sqrt(variance)
    return mean, std


def load_image_paths_from_folder(root_dir, class_names):
    """폴더 구조를 탐색하여 이미지 경로를 가져옴."""
    image_paths = []
    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(class_dir, file))
    return image_paths


def save_results(mean, std, global_min_size, output_dir):
    """Mean, std, and global min size 저장"""
    os.makedirs(output_dir, exist_ok=True)

    # mean과 std 저장
    stats_file = os.path.join(output_dir, "dataset_stats.txt")
    with open(stats_file, "w") as f:
        f.write(f"Dataset mean: {mean.tolist()}\n")
        f.write(f"Dataset std: {std.tolist()}\n")
    print(f"Dataset mean and std saved to {stats_file}")

    # global min size 저장
    min_size_file = os.path.join(output_dir, "global_min_size.txt")
    with open(min_size_file, "w") as f:
        f.write(f"Global Min Height: {global_min_size[0]}\n")
        f.write(f"Global Min Width: {global_min_size[1]}\n")
    print(f"Global min size saved to {min_size_file}")


def main():
    # train_x 폴더와 클래스 이름 설정
    train_x_dir = "./train_x"
    class_names = ["can_1", "can_2", "can_3", "cup_1", "cup_2", "cup_3", "mug_1", "mug_2", "mug_3", "bottle_1", "bottle_2", "bottle_3"]

    # 이미지 경로 로드
    all_image_paths = load_image_paths_from_folder(train_x_dir, class_names)

    # 전체 데이터셋에서 mean과 std 계산
    mean, std = calculate_mean_std(all_image_paths)
    print(f"Dataset mean: {mean}, std: {std}")

    # 전체 데이터셋에서 최소 크기 계산
    global_min_size = find_global_min_size(all_image_paths)
    print(f"Global resizing dimensions (height, width): {global_min_size}")

    # 결과 저장
    output_dir = "./outputs"
    save_results(mean, std, global_min_size, output_dir)


if __name__ == "__main__":
    main()
