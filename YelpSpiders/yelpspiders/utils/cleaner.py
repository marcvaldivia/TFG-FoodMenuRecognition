import json
import logging
import os
import urllib
import shutil

import requests

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


if __name__ == '__main__':
    s = Cleaner(Path.DATA_FOLDER)
    s.remove_empty_folders(Path.DATA_FOLDER)
