import json
import logging
import os
import urllib

import requests

from yelpspiders.variables.paths import Path


class Downloader:

    def __init__(self, root_folder, overwrite=False):
        self.root_folder = root_folder
        self.overwrite = overwrite
        logging.info("Starting downloader...")

    def execute(self):
        # DataSet folders of the different restaurants
        folders = [o for o in os.listdir(self.root_folder)
                   if os.path.isdir("%s/%s" % (self.root_folder, o)) and o != "data"]
        for f in folders:
            # Load JSON file with the restaurant information
            f = "%s/%s" % (self.root_folder, f)
            data = json.load(open('%s/info.json' % f))
            self.execute_restaurant(f, data)

    def execute_restaurant(self, f, data):
        try:
            # Get the different menus of the current restaurant
            for m in data['menus']:
                menu_name = m.keys()[0]  # Name of the menu
                directory = "%s/%s" % (f, menu_name)
                # Creates a directory to store the menu files
                if not os.path.exists(directory):
                    os.makedirs(directory)
                # Get the different dishes of the menu grouped by headers
                for k in m[menu_name]:  # 'k' is a dictionary with the dishes for every header
                    headers = k.keys()
                    for h in headers:
                        # Get the different dishes
                        for dish in k[h]:
                            # Creates a directory for every dish in the menu
                            dish_directory = "%s/%s" % (directory, dish)
                            if not os.path.exists(dish_directory):
                                os.makedirs(dish_directory)
                            logging.info("Dish directory: %s" % dish_directory)
                            logging.info("Dish name: %s" % dish)
                            self.execute_download(k, h, dish, dish_directory)
        except Exception as ex:
            logging.error(str(ex))

    @staticmethod
    def call_to_api(img):
        url = 'http://logmeal.ml:8088/api/v0.6/foodRecognitionMultipleUrl'
        img_urls = [img]
        data = dict()
        data['urls'] = img_urls
        data['numberFoods'] = -1
        headers = {'Content-Type': 'application/json'}
        req = requests.post(url, json=data, headers=headers)
        return req

    def execute_download(self, k, h, dish, dish_directory):
        # Get all the URL images for every dish
        for url in k[h][dish]:
            names = url.split("/")
            name = names[len(names)-2]
            logging.info("Dish name from URL: %s" % dish)
            path_to_write = "%s/%s" % (dish_directory, name)
            # Generate or update the image and JSON file according to the API response
            if not os.path.exists("%s.json" % path_to_write) or self.overwrite:
                logging.info("Dish image URL: %s" % url)
                print("Dish image URL: %s" % url)
                print(path_to_write)
                # r = requests.get("http://logmeal.ml:8088/api/v0.5/complete?img_url=%s" % url)
                r = self.call_to_api(url)
                # If the response is positive, we will download the image and write the answer
                if r.status_code == 200:
                    if not os.path.exists("%s.jpg" % path_to_write):
                        urllib.urlretrieve(url, "%s.jpg" % path_to_write)
                    with open("%s.json" % path_to_write, "w") as json_file:
                        json_file.write(r.text)

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
        if len(files) == 0 and remove_root:
            print "Removing empty folder:", path
            os.rmdir(path)


if __name__ == '__main__':
    s = Downloader(Path.DATA_FOLDER)
    s.remove_empty_folders(Path.DATA_FOLDER)
    s.execute()
    s.remove_empty_folders(Path.DATA_FOLDER)
