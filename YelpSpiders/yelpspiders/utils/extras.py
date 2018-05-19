import glob
import logging
import os
import os.path
import shutil

from yelpspiders.variables.paths import Path


class Cleaner:

    def __init__(self, root_folder):
        self.root_folder = root_folder
        logging.info("Starting cleaner...")

    def remove_empty_folders(self, path, remove_root=True):
        if not os.path.isdir(path):
            return
        # remove empty sub-folders
        files = os.listdir(path)
        if len(files):
            for f in files:
                full_path = os.path.join(path, f)
                if os.path.isdir(full_path):
                    self.remove_empty_folders(full_path)

        # if folder empty, delete it
        files = os.listdir(path)
        if "info.json" in files and len(files) == 1 and remove_root:
            print("Removing empty folder:", path)
            shutil.rmtree(path)


class Info:

    def __init__(self, root_folder):
        self.root_folder = root_folder

    def get_images(self):
        return len(glob.glob(self.root_folder + '/*/*/*/*.jpg'))

    def get_restaurants(self):
        files_depth1 = glob.glob(self.root_folder + '/*')
        dirs_depth1 = filter(lambda f: os.path.isdir(f), files_depth1)
        return len(dirs_depth1) - 1

    def get_dishes(self):
        files_depth3 = glob.glob(self.root_folder + '/*/*/*')
        dirs_depth3 = filter(lambda f: os.path.isdir(f), files_depth3)
        return len(dirs_depth3)


if __name__ == '__main__':
    # s = Cleaner(Path.DATA_FOLDER)
    # s.remove_empty_folders(Path.DATA_FOLDER)
    info = Info(Path.DATA_FOLDER)
    print("Number of images: %s" % info.get_images())
    print("Number of restaurants: %s" % info.get_restaurants())
    print("Number of dishes: %s" % info.get_dishes())
