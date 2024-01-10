import os
import shutil

# 定义文件夹路径
label_data_folder = './data'
train_folder = './data/train'
test_folder = './data/test'
library_folder = './data/library'

# 划分数据集函数
def split_dataset():
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    folders = sorted(os.listdir(label_data_folder), key=lambda x: int(x))
    total_folders = len(folders)
    train_count = int(total_folders * 0.8)

    train_folders = folders[:train_count]
    test_folders = folders[train_count:]

    def move_folders(src_folder, dst_folder):
        os.makedirs(dst_folder, exist_ok=True)
        for folder_name in src_folder:
            src_path = os.path.join(label_data_folder, folder_name)
            dst_path = os.path.join(dst_folder, folder_name)
            shutil.move(src_path, dst_path)

    move_folders(train_folders, train_folder)
    move_folders(test_folders, test_folder)

    print("数据集划分完成！")

# 创建library并移动图片函数
def create_library():
    test_folders = sorted(os.listdir(test_folder), key=lambda x: int(x))

    os.makedirs(library_folder, exist_ok=True)

    for folder_name in test_folders:
        test_subfolder = os.path.join(test_folder, folder_name)
        library_subfolder = os.path.join(library_folder, folder_name)

        os.makedirs(library_subfolder, exist_ok=True)

        images = os.listdir(test_subfolder)
        if len(images) > 0:
            image_to_move = images[0]
            src_image = os.path.join(test_subfolder, image_to_move)
            dst_image = os.path.join(library_subfolder, image_to_move)
            shutil.move(src_image, dst_image)

    print("library文件夹创建并图片移动完成！")

# 执行数据集划分
split_dataset()

# 创建library并移动图片
create_library()