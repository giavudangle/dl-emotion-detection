import os


def list_file(path):
    for root, dirs, files in os.walk(path, topdown=True):
        for name in files:
            if root.split('\\')[-1] != "Tỉ lệ dữ liệu" and (name[-3:] == "jpg" or name[-3:] == "png"):
                yield os.path.join(root, name)
