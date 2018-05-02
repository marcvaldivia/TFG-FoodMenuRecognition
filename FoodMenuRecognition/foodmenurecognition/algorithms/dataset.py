import json
import logging
import os
import random

import numpy as np
from sklearn.model_selection import train_test_split

from foodmenurecognition.variables.paths import Path


class DataSet:

    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.result_ingredients, self.idx_ingredients = dict(), -1
        self.result_recognition, self.idx_recognition = dict(), -1
        self.result_family, self.idx_family = dict(), -1
        self.folders = [o for o in os.listdir(self.root_folder)
                        if os.path.isdir("%s/%s" % (self.root_folder, o))]
        self.train_set, self.val_set, self.test_set = list(), list(), list()
        self.generate_json(split_kind=2)
        self.all_set = self.train_set + self.val_set + self.test_set
        self.create_dictionaries()
        np.save("%s/data/ingredients.npy" % self.root_folder, self.result_ingredients)
        np.save("%s/data/recognition.npy" % self.root_folder, self.result_recognition)
        np.save("%s/data/family.npy" % self.root_folder, self.result_family)
        print("Ingredients: %s" % len(self.result_ingredients))
        print("Recognition: %s" % len(self.result_recognition))
        print("Family: %s" % len(self.result_family))

    def execute(self):
        for x in self.all_set:
            json_file = x[1]
            vec_ingredients = np.zeros(self.idx_ingredients + 1, dtype=float)
            vec_recognition = np.zeros(self.idx_recognition + 1, dtype=float)
            vec_family = np.zeros(self.idx_family + 1, dtype=float)
            for ingredient in json_file['result_ingredients']:
                vec_ingredients[self.result_ingredients[ingredient['class']]] = ingredient['prob']
            for recognition in json_file['result_recognition']:
                vec_recognition[self.result_recognition[recognition['class']]] = recognition['prob']
            for family in json_file['result_family']:
                vec_family[self.result_family[family['class']]] = family['prob']
            np.save("%s.npy" % x[0], np.concatenate((vec_ingredients, vec_recognition, vec_family)))

    def execute_files(self, name):
        dishes = open("%s/data/dishes_%s.txt" % (self.root_folder, name), 'w')
        links = open("%s/data/links_%s.txt" % (self.root_folder, name), 'w')
        outs = open("%s/data/outs_%s.txt" % (self.root_folder, name), 'w')
        my_dishes = set()
        for x in self.train_set if name == 'train' else self.val_set if name == 'val' else self.test_set:
            link, dish = x[0].replace(Path.DATA_FOLDER, ""), x[2]
            my_dishes.add(dish)
            dishes.write("%s\n" % dish)
            links.write("%s.npy\n" % link)
            outs.write("1\n")
        if name == 'train':
            my_dishes = list(my_dishes)
            for x in self.train_set:
                link, dish = x[0].replace(Path.DATA_FOLDER, ""), x[2]
                dishes.write("%s\n" % random.choice(my_dishes))
                links.write("%s.npy\n" % link)
                outs.write("0\n")
            dishes.close()
            links.close()
            outs.close()

    def create_dictionaries(self):
        dishes = self.all_set
        for d in dishes:
            json_file = d[1]
            for ingredient in json_file['result_ingredients']:
                self.idx_ingredients = self.dictionary_el(self.result_ingredients, self.idx_ingredients,
                                                          ingredient['class'])
            for recognition in json_file['result_recognition']:
                self.idx_recognition = self.dictionary_el(self.result_recognition, self.idx_recognition,
                                                          recognition['class'])
            for family in json_file['result_family']:
                self.idx_family = self.dictionary_el(self.result_family, self.idx_family, family['class'])

    @staticmethod
    def dictionary_el(dictionary, idx, el):
        try:
            _ = dictionary[el]
        except:
            idx += 1
            dictionary[el] = idx
        return idx

    def decide_where_to_add(self, l1, l2, l3, split_kind, json_f, json_object, d):
        if split_kind == 0:
            if l1 == 0:
                self.train_set.append((json_f.replace(".json", ""), json_object, d))
            elif l1 == 1:
                self.val_set.append((json_f.replace(".json", ""), json_object, d))
            elif l1 == 2:
                self.test_set.append((json_f.replace(".json", ""), json_object, d))
        elif split_kind == 1:
            if l2 == 0:
                self.train_set.append((json_f.replace(".json", ""), json_object, d))
            elif l2 == 1:
                self.val_set.append((json_f.replace(".json", ""), json_object, d))
            elif l2 == 2:
                self.test_set.append((json_f.replace(".json", ""), json_object, d))
        elif split_kind == 2:
            if l3 == 0:
                self.train_set.append((json_f.replace(".json", ""), json_object, d))
            elif l3 == 1:
                self.val_set.append((json_f.replace(".json", ""), json_object, d))
            elif l3 == 2:
                self.test_set.append((json_f.replace(".json", ""), json_object, d))

    def generate_json(self, split_kind=0):
        random.shuffle(self.folders)
        my_folders = [self.folders]
        if split_kind == 0:
            train, other = train_test_split(self.folders, test_size=0.4)
            val, test = train_test_split(other, test_size=0.7)
            my_folders = [train] + [val] + [test]
        for l1, folds in enumerate(my_folders):
            for f in folds:
                sub_folders = [o for o in os.listdir("%s/%s" % (self.root_folder, f))
                               if os.path.isdir("%s/%s/%s" % (self.root_folder, f, o))]
                for sub in sub_folders:
                    dishes = [o for o in os.listdir("%s/%s/%s" % (self.root_folder, f, sub))
                              if os.path.isdir("%s/%s/%s/%s" % (self.root_folder, f, sub, o))]
                    random.shuffle(dishes)
                    my_dishes = [dishes]
                    if split_kind == 1:
                        train, other = train_test_split(dishes, test_size=0.4)
                        val, test = train_test_split(other, test_size=0.7)
                        my_dishes = [train] + [val] + [test]
                    for l2, ds in enumerate(my_dishes):
                        for d in ds:
                            json_files = [j for j in os.listdir("%s/%s/%s/%s" % (self.root_folder, f, sub, d))
                                          if j.endswith('.json')]
                            random.shuffle(json_files)
                            my_json = [json_files]
                            if split_kind == 2:
                                train, other = train_test_split(json_files, test_size=0.4)
                                val, test = train_test_split(other, test_size=0.7)
                                my_json = [train] + [val] + [test]
                            for l3, j_son in enumerate(my_json):
                                for json_f in j_son:
                                    json_f = "%s/%s/%s/%s/%s" % (self.root_folder, f, sub, d, json_f)
                                    try:
                                        json_object = json.load(open(json_f))
                                        if len(json_files) > 4:
                                            self.decide_where_to_add(l1, l2, l3, split_kind, json_f, json_object, d)
                                        else:
                                            self.train_set.append((json_f.replace(".json", ""), json_object, d))
                                    except:
                                        logging.error("Error loading JSON file %s" % json_f)


if __name__ == '__main__':
    s = DataSet(Path.DATA_FOLDER)
    s.execute()
    s.execute_files('train')
    s.execute_files('val')
    s.execute_files('test')
