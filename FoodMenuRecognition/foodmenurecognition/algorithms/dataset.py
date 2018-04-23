import os
import json
import numpy as np
import random
import math

from foodmenurecognition.variables.paths import Path


class DataSet:

    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.result_ingredients, self.idx_ingredients = dict(), -1
        self.result_recognition, self.idx_recognition = dict(), -1
        self.result_family, self.idx_family = dict(), -1
        self.folders = [o for o in os.listdir(self.root_folder)
                        if os.path.isdir("%s/%s" % (self.root_folder, o))]
        random.shuffle(self.folders)
        self.train = int(math.ceil(len(self.folders)*0.7))
        self.f_train = self.folders[:self.train]
        self.val = int(math.floor(len(self.folders)*0.1))
        self.f_val = self.folders[self.train:self.train+self.val]
        self.test = len(self.folders) - (self.train + self.val)
        self.f_test = self.folders[self.train+self.val:]
        self.create_dictionaries()
        np.save("%s/data/ingredients.npy" % self.root_folder, self.result_ingredients)
        np.save("%s/data/recognition.npy" % self.root_folder, self.result_recognition)
        np.save("%s/data/family.npy" % self.root_folder, self.result_family)
        print("Ingredients: %s" % len(self.result_ingredients))
        print("Recognition: %s" % len(self.result_recognition))
        print("Family: %s" % len(self.result_family))

    def execute(self):
        for x in self.generate_json(self.f_train+self.f_val+self.f_test):
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

    def execute_files(self, folders, name):
        dishes = open("%s/data/dishes_%s.txt" % (self.root_folder, name), 'w')
        links = open("%s/data/links_%s.txt" % (self.root_folder, name), 'w')
        outs = open("%s/data/outs_%s.txt" % (self.root_folder, name), 'w')
        my_dishes = set()
        for x in self.generate_json(folders):
            link, dish = x[0].replace("/Users/yoda/git/TFG-FoodMenuRecognition/dataset/", ""), x[2]
            my_dishes.add(dish)
            dishes.write("%s\n" % dish)
            links.write("%s.npy\n" % link)
            outs.write("1\n")
        if name == 'train':
            my_dishes = list(my_dishes)
            for x in self.generate_json(folders):
                link, dish = x[0].replace("/Users/yoda/git/TFG-FoodMenuRecognition/dataset/", ""), x[2]
                dishes.write("%s\n" % random.choice(my_dishes))
                links.write("%s.npy\n" % link)
                outs.write("0\n")
            dishes.close()
            links.close()
            outs.close()

    def create_dictionaries(self):
        dishes = self.generate_json(self.f_train+self.f_val+self.f_test)
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

    def generate_json(self, folders):
        for f in folders:
            sub_folders = (o for o in os.listdir("%s/%s" % (self.root_folder, f))
                           if os.path.isdir("%s/%s/%s" % (self.root_folder, f, o)))
            for sub in sub_folders:
                dishes = (o for o in os.listdir("%s/%s/%s" % (self.root_folder, f, sub))
                          if os.path.isdir("%s/%s/%s/%s" % (self.root_folder, f, sub, o)))
                for d in dishes:
                    json_files = (j for j in os.listdir("%s/%s/%s/%s" % (self.root_folder, f, sub, d))
                                  if j.endswith('.json'))
                    for json_f in json_files:
                        json_f = "%s/%s/%s/%s/%s" % (self.root_folder, f, sub, d, json_f)
                        json_object = json.load(open(json_f))
                        yield (json_f.replace(".json", ""), json_object, d)


if __name__ == '__main__':
    s = DataSet(Path.DATA_FOLDER)
    s.execute()
    s.execute_files(s.f_train, 'train')
    s.execute_files(s.f_val, 'val')
    s.execute_files(s.f_test, 'test')
