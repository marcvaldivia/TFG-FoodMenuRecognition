#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
import random
import glob

import numpy as np
from sklearn.model_selection import train_test_split

from foodmenurecognition.variables.paths import Path


class DataSet:

    def __init__(self, root_folder, split_kind=2, ingredients=False, min_els=5):
        self.root_folder = root_folder
        self.split_kind = split_kind
        self.ingredients = ingredients
        self.min_els = min_els
        self.result_ingredients, self.idx_ingredients = dict(), -1
        self.result_recognition, self.idx_recognition = dict(), -1
        self.result_family, self.idx_family = dict(), -1
        self.restaurants = [o for o in os.listdir(self.root_folder)
                            if os.path.isdir("%s/%s" % (self.root_folder, o)) and o != 'data']
        self.all_set, self.dishes, self.images = list(), set(), set()
        self.train, self.val, self.test = list(), list(), list()
        self.generate_json()
        self.create_dictionaries()
        self.total, self.total_dishes = len(self.all_set), len(self.dishes)
        np.save("%s/data/ingredients.npy" % self.root_folder, self.result_ingredients)
        np.save("%s/data/recognition.npy" % self.root_folder, self.result_recognition)
        np.save("%s/data/family.npy" % self.root_folder, self.result_family)
        print("Ingredients: %s" % len(self.result_ingredients))
        print("Recognition: %s" % len(self.result_recognition))
        print("Family: %s" % len(self.result_family))

    @staticmethod
    def get_new_version_api(json_file, name):
        dict_ret = list()
        json_file = json_file['results'][0]
        to_enum = json_file[name]['tops'] if name != 'ingredients' else json_file[name][name]
        probabilities = json_file[name]['probs']
        for idx, top in enumerate(to_enum):
            dict_ret.append({"class": top, "prob": probabilities[idx]})
        return dict_ret

    def execute(self):
        for x in self.all_set:
            try:
                json_file = x[1]
                vec_ingredients = np.zeros(self.idx_ingredients + 1, dtype=float)
                vec_recognition = np.zeros(self.idx_recognition + 1, dtype=float)
                vec_family = np.zeros(self.idx_family + 1, dtype=float)
                for ingredient in self.get_new_version_api(json_file, 'ingredients'):
                    vec_ingredients[self.result_ingredients[ingredient['class']]] = ingredient['prob']
                for recognition in self.get_new_version_api(json_file, 'foodRecognition'):
                    vec_recognition[self.result_recognition[recognition['class']]] = recognition['prob']
                for family in self.get_new_version_api(json_file, 'foodFamily'):
                    vec_family[self.result_family[family['class']]] = family['prob']
                if self.ingredients:
                    np.save("%s.npy" % x[0], np.concatenate((vec_ingredients, vec_recognition, vec_family)))
                else:
                    np.save("%s.npy" % x[0], np.concatenate((vec_recognition, vec_family)))
            except Exception as ex:
                logging.error(ex)
                logging.error(x)
                # os.remove("%s.jpg" % x[0])
                # os.remove("%s.json" % x[0])

    def execute_files(self, name):
        dishes = open("%s/data/dishes_%s.txt" % (self.root_folder, name), 'w')
        links = open("%s/data/links_%s.txt" % (self.root_folder, name), 'w')
        cnn = open("%s/data/cnn_%s.txt" % (self.root_folder, name), 'w')
        outs = open("%s/data/outs_%s.txt" % (self.root_folder, name), 'w')
        my_dishes = set()
        for x in self.train if name == 'train' else self.val if name == 'val' else self.test:
            link, dish = x[0].replace(Path.DATA_FOLDER, ""), x[2]
            if os.path.exists(Path.DATA_FOLDER + link + "_cnn.npy"):
                my_dishes.add(dish)
                dishes.write("%s\n" % dish)
                links.write("%s.npy\n" % link)
                cnn.write("%s_cnn.npy\n" % link)
                outs.write("1\n")
        if name == 'train':
            for x in self.train:
                link, dish = x[0].replace(Path.DATA_FOLDER, ""), x[2]
                if os.path.exists(link + "_cnn.npy"):
                    segments = link.split("/")
                    food_dir = segments[:-2]
                    count = 0
                    d = Path.DATA_FOLDER + "/" + food_dir[-2] + "/" + food_dir[-1]
                    all_foods = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
                    # d = Path.DATA_FOLDER + "/" + food_dir[-2]
                    # files_depth2 = glob.glob('%s/*/*' % d)
                    # all_foods = filter(lambda f: os.path.isdir(f), files_depth2)
                    for food in [random.choice(all_foods)]:
                        if food != dish:
                            count += 1
                            food_name = food.split("/")[-1]
                            dishes.write("%s\n" % food_name)
                            links.write("%s.npy\n" % link)
                            cnn.write("%s_cnn.npy\n" % link)
                            outs.write("0\n")
            dishes.close()
            links.close()
            outs.close()

    def create_dictionaries(self):
        for d in self.all_set:
            try:
                json_file = d[1]
                for ingredient in self.get_new_version_api(json_file, 'ingredients'):
                    self.idx_ingredients = self.dictionary_el(self.result_ingredients, self.idx_ingredients,
                                                              ingredient['class'])
                for recognition in self.get_new_version_api(json_file, 'foodRecognition'):
                    self.idx_recognition = self.dictionary_el(self.result_recognition, self.idx_recognition,
                                                              recognition['class'])
                for family in self.get_new_version_api(json_file, 'foodFamily'):
                    self.idx_family = self.dictionary_el(self.result_family, self.idx_family, family['class'])
            except Exception as ex:
                logging.error(ex)
                logging.error(d)

    @staticmethod
    def dictionary_el(dictionary, idx, el):
        try:
            _ = dictionary[el]
        except:
            idx += 1
            dictionary[el] = idx
        return idx

    def generate_json(self):
        for restaurant in self.restaurants:
            sub_folders = [o for o in os.listdir("%s/%s" % (self.root_folder, restaurant))
                           if os.path.isdir("%s/%s/%s" % (self.root_folder, restaurant, o))]
            for menu in sub_folders:
                dishes = [o for o in os.listdir("%s/%s/%s" % (self.root_folder, restaurant, menu))
                          if os.path.isdir("%s/%s/%s/%s" % (self.root_folder, restaurant, menu, o))]
                for dish in dishes:
                    self.dishes.add(dish)
                    json_files = [j for j in os.listdir("%s/%s/%s/%s" % (self.root_folder, restaurant, menu, dish))
                                  if j.endswith('.json')]
                    for image in json_files:
                        self.images.add(image.replace(".json", ""))
                        try:
                            image = "%s/%s/%s/%s/%s" % (self.root_folder, restaurant, menu, dish, image)
                            json_object = json.load(open(image))
                            self.all_set.append((image.replace(".json", ""), json_object, dish,
                                                 len(dishes), len(json_files)))
                        except Exception as ex:
                            logging.error(ex)
                            logging.error("Error loading JSON file %s" % image)

    def execute_division(self):
        for img in self.all_set:
            if img[-1] < self.min_els or img[-2] < self.min_els:
                self.train.append(img)
                self.all_set.remove(img)
                if self.split_kind == 1:
                    self.dishes.remove(img[-3])
        if self.split_kind == 0:
            train = 0.7
            tmp_train, tmp_test = train_test_split(self.restaurants, train_size=train)
            for x in tmp_train:
                self.restaurants.remove(x)
            tmp_train = [x for x in self.all_set if x[0].split("/")[-4] in tmp_train]
            tmp_val, tmp_test = train_test_split(self.restaurants, train_size=0.33)
            tmp_val = [x for x in self.all_set if x[0].split("/")[-4] in tmp_val]
            tmp_test = [x for x in self.all_set if x[0].split("/")[-4] in tmp_test]
        elif self.split_kind == 1:
            train = 0.7 - (len(self.dishes)*1.0 / self.total_dishes)
            tmp_train, tmp_test = train_test_split(self.dishes, train_size=train)
            for x in tmp_train:
                self.dishes.remove(x)
            tmp_train = [x for x in self.all_set if x[0].split("/")[-2] in tmp_train]
            tmp_val, tmp_test = train_test_split(self.restaurants, train_size=0.33)
            tmp_val = [x for x in self.all_set if x[0].split("/")[-2] in tmp_val]
            tmp_test = [x for x in self.all_set if x[0].split("/")[-2] in tmp_test]
        else:
            train = 0.7 - (len(self.train)*1.0 / self.total)
            tmp_train, tmp_test = train_test_split(self.all_set, train_size=train)
            for x in tmp_train:
                self.all_set.remove(x)
            tmp_train, tmp_test = train_test_split(self.all_set, train_size=train)
            tmp_val, tmp_test = train_test_split(self.all_set, train_size=0.33)
        self.train, self.val, self.test = tmp_train, tmp_val, tmp_test


if __name__ == '__main__':
    s = DataSet(Path.DATA_FOLDER)
    s.execute()
    s.execute_division()
    s.execute_files('train')
    s.execute_files('val')
    s.execute_files('test')

