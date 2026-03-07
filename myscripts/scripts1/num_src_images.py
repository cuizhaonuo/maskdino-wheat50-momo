import os

root_dir = "/public/lab_share/nfs_field/raw_data/DJI/P1/zhaoxian2023/wheat/20240516DJI/DCIM"

for folder in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder)

    if os.path.isdir(folder_path):
        jpg_count = sum(
            1 for f in os.listdir(folder_path)
            if f.lower().endswith(".jpg")
        )

        print(f"{folder}: {jpg_count} JPG files")