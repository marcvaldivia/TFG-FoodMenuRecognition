import glob,os.path
from keras_wrapper.dataset import Dataset, saveDataset, loadDataset

from foodmenurecognition.variables.paths import Path


def build_dataset(params):
    if params['REBUILD_DATASET']:

        base_path = params['DATA_ROOT_PATH']
        name = params['DATASET_NAME']
        ds = Dataset(name, base_path, silence=False)

        # INPUT DATA
        ds.setInput(base_path + '/' + params['DISHES_FILES']['train'],
                    'train',
                    type='text',
                    id=params['INPUTS_IDS_DATASET'][1],
                    build_vocabulary=True,
                    tokenization=params['TOKENIZATION_METHOD'],
                    fill=params['FILL'],
                    pad_on_batch=True,
                    max_text_len=params['MAX_OUTPUT_TEXT_LEN'],
                    min_occ=params['MIN_OCCURRENCES_VOCAB'])

        ds.setInput(base_path + '/' + params['DISHES_FILES']['val'],
                    'val',
                    type='text',
                    id=params['INPUTS_IDS_DATASET'][1],
                    build_vocabulary=True,
                    pad_on_batch=True,
                    tokenization=params['TOKENIZATION_METHOD'],
                    max_text_len=params['MAX_OUTPUT_TEXT_LEN_TEST'],
                    min_occ=params['MIN_OCCURRENCES_VOCAB'])

        ds.setInput(base_path + '/' + params['DISHES_FILES']['test'],
                    'test',
                    type='text',
                    id=params['INPUTS_IDS_DATASET'][1],
                    build_vocabulary=True,
                    pad_on_batch=True,
                    tokenization=params['TOKENIZATION_METHOD'],
                    max_text_len=params['MAX_OUTPUT_TEXT_LEN_TEST'],
                    min_occ=params['MIN_OCCURRENCES_VOCAB'])

        # INPUT DATA
        ds.setInput(base_path + '/' + params['IMAGES_LIST_FILES']['train'],
                    'train',
                    type='image-features',
                    id=params['INPUTS_IDS_DATASET'][0],
                    feat_len=params['IMG_FEAT_SIZE'])

        ds.setInput(base_path + '/' + params['IMAGES_LIST_FILES']['val'],
                    'val',
                    type='image-features',
                    id=params['INPUTS_IDS_DATASET'][0],
                    feat_len=params['IMG_FEAT_SIZE'])

        ds.setInput(base_path + '/' + params['IMAGES_LIST_FILES']['test'],
                    'test',
                    type='image-features',
                    id=params['INPUTS_IDS_DATASET'][0],
                    feat_len=params['IMG_FEAT_SIZE'])

        # OUTPUT DATA
        ds.setOutput(base_path + '/' + params['OUT_FILES']['train'],
                     'train',
                     type='real',
                     id=params['OUTPUTS_IDS_DATASET'][0])

        ds.setOutput(base_path + '/' + params['OUT_FILES']['val'],
                     'val',
                     type='real',
                     id=params['OUTPUTS_IDS_DATASET'][0])

        ds.setOutput(base_path + '/' + params['OUT_FILES']['test'],
                     'test',
                     type='real',
                     id=params['OUTPUTS_IDS_DATASET'][0])  # TODO: Pot ser el array directament

        # TODO: Afegir array de pesos
        # ds.sample_weights[params['OUTPUTS_IDS_DATASET'][0]] = None

        # We have finished loading the dataset, now we can store it for using it in the future
        saveDataset(ds, params['DATASET_STORE_PATH'])
    else:
        # We can easily recover it with a single line
        ds = loadDataset(params['DATASET_STORE_PATH'] + '/Dataset_' + params['DATASET_NAME'] + '.pkl')
    return ds


def _build_dataset_test(params):
    base_path = params['DATA_ROOT_PATH']
    name = params['DATASET_NAME']
    ds = Dataset(name, base_path, silence=False)

    # INPUT DATA
    ds.setInput(base_path + '/data/new_dishes_val.txt',
                'val',
                type='text',
                id=params['INPUTS_IDS_DATASET'][1],
                build_vocabulary=True,
                pad_on_batch=True,
                tokenization=params['TOKENIZATION_METHOD'],
                max_text_len=params['MAX_OUTPUT_TEXT_LEN_TEST'],
                min_occ=params['MIN_OCCURRENCES_VOCAB'])

    ds.setInput(base_path + '/data/new_dishes_test.txt',
                'test',
                type='text',
                id=params['INPUTS_IDS_DATASET'][1],
                build_vocabulary=True,
                pad_on_batch=True,
                tokenization=params['TOKENIZATION_METHOD'],
                max_text_len=params['MAX_OUTPUT_TEXT_LEN_TEST'],
                min_occ=params['MIN_OCCURRENCES_VOCAB'])

    # INPUT DATA
    ds.setInput(base_path + '/data/new_links_val.txt',
                'val',
                type='image-features',
                id=params['INPUTS_IDS_DATASET'][0],
                feat_len=params['IMG_FEAT_SIZE'])

    ds.setInput(base_path + '/data/new_links_test.txt',
                'test',
                type='image-features',
                id=params['INPUTS_IDS_DATASET'][0],
                feat_len=params['IMG_FEAT_SIZE'])

    # OUTPUT DATA
    ds.setOutput(base_path + '/data/new_outs_val.txt',
                 'val',
                 type='real',
                 id=params['OUTPUTS_IDS_DATASET'][0])

    ds.setOutput(base_path + '/data/new_outs_test.txt',
                 'test',
                 type='real',
                 id=params['OUTPUTS_IDS_DATASET'][0])  # TODO: Pot ser el array directament

    # TODO: Afegir array de pesos
    # ds.sample_weights[params['OUTPUTS_IDS_DATASET'][0]] = None
    return ds


def build_dataset_test(params, name):
    base_path = params['DATA_ROOT_PATH']
    links = open(base_path + '/' + params['IMAGES_LIST_FILES'][name], 'r')
    new_dishes = open("%s/data/new_dishes_%s.txt" % (Path.DATA_FOLDER, name), 'w')
    new_links = open("%s/data/new_links_%s.txt" % (Path.DATA_FOLDER, name), 'w')
    new_outs = open("%s/data/new_outs_%s.txt" % (Path.DATA_FOLDER, name), 'w')
    index = open("%s/data/index_%s.txt" % (Path.DATA_FOLDER, name), 'w')
    l_content = [x.strip() for x in links.readlines()]
    for link in l_content:
        segments = link.split("/")
        food_dir = segments[:-2]
        count = 0
        d = Path.DATA_FOLDER + "/" + food_dir[-2] + "/" + food_dir[-1]
        all_foods = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
        #d = Path.DATA_FOLDER + "/" + food_dir[-2]
        #files_depth2 = glob.glob('%s/*/*' % d)
        #all_foods = filter(lambda f: os.path.isdir(f), files_depth2)
        if len(all_foods) > 2:
            for food in all_foods:
                count += 1
                food_name = food.split("/")[-1]
                new_dishes.write("%s\n" % food_name)
                new_links.write("%s\n" % link)
                new_outs.write("%s\n" % ("0" if food_name != segments[-2] else "1"))
            index.write("%s\n" % count)
    new_dishes.close()
    new_links.close()
    new_outs.close()
    index.close()
