filename = '/public/home/zncui/pycharm_project/MaskDINO/myscripts/processed_image_names.txt'
def check_filename_is_unique(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    filenames = [line.strip() for line in lines]
    unique_filenames = set(filenames)
    if len(filenames) != len(unique_filenames):
        print("There are duplicate filenames in the list.")
        return False
    else:
        print("All filenames are unique.")
        return True
check_filename_is_unique(filename)