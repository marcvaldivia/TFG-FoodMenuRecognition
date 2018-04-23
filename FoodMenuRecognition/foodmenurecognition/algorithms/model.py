import logging
from timeit import default_timer as timer

import numpy as np
from keras_wrapper.cnn_model import loadModel

from foodmenurecognition.conf.config import load_parameters
from foodmenurecognition.models.fooddesc_model import FoodDesc_Model
from foodmenurecognition.utils.prepare_data import build_dataset, build_dataset_test, _build_dataset_test
from foodmenurecognition.variables.paths import Path

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def train_model(params):
    # Load data
    dataset = build_dataset(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][1]]

    # Build model
    food_model = FoodDesc_Model(params,
                                type=params['MODEL_TYPE'],
                                verbose=params['VERBOSE'],
                                model_name=params['MODEL_NAME'] + '_reloaded',
                                vocabularies=dataset.vocabulary,
                                store_path=params['STORE_PATH'],
                                set_optimizer=False,
                                clear_dirs=False)

    # Define the inputs and outputs mapping from our Dataset instance to our model
    inputMapping = dict()
    for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
        if len(food_model.ids_inputs) > i:
            pos_source = dataset.ids_inputs.index(id_in)
            id_dest = food_model.ids_inputs[i]
            inputMapping[id_dest] = pos_source
    food_model.setInputsMapping(inputMapping)

    outputMapping = dict()
    for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
        if len(food_model.ids_outputs) > i:
            pos_target = dataset.ids_outputs.index(id_out)
            id_dest = food_model.ids_outputs[i]
            outputMapping[id_dest] = pos_target
    food_model.setOutputsMapping(outputMapping)

    food_model.setOptimizer()
    params['MAX_EPOCH'] += params['RELOAD']

    # Update optimizer either if we are loading or building a model
    food_model.params = params
    food_model.setOptimizer()

    total_start_time = timer()

    logger.debug('Starting training!')
    training_params = {
        'normalize': False,
        'n_epochs': 10,
        'batch_size': 64,
        'n_parallel_loaders': 1
    }

    food_model.trainNet(dataset, training_params)

    total_end_time = timer()
    time_difference = total_end_time - total_start_time
    logging.info('In total is {0:.2f}s = {1:.2f}m'.format(time_difference, time_difference / 60.0))


def apply_food_model(params):
    # Load data
    dataset = build_dataset(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][1]]

    # Load model
    food_model = loadModel(params['STORE_PATH'], 1)
    food_model.setOptimizer()

    for s in params["EVAL_ON_SETS2"]:
        # Apply model predictions
        params_prediction = {
            'predict_on_sets': [s],
            'normalize': False,
            'n_parallel_loaders': 1
        }

        predictions = food_model.predictNet(dataset, params_prediction)[s]

        with open("test_sampling.pred", "w") as pred:
            for p in predictions:
                pred.write("%s\n" % np.array2string(p))


def test_data(params):
    build_dataset_test(params, 'val')
    build_dataset_test(params, 'test')
    dataset = _build_dataset_test(params)

    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][1]]  # Load model
    food_model = loadModel(params['STORE_PATH'], 10)
    food_model.setOptimizer()

    for s in params["EVAL_ON_SETS"]:
        # Apply model predictions
        params_prediction = {
            'predict_on_sets': [s],
            'normalize': False,
            'n_parallel_loaders': 1
        }
        predictions = food_model.predictNet(dataset, params_prediction)[s]

        total, top1, top2, top5, top7, top10 = 0, 0, 0, 0, 0, 0
        index = open("%s/data/index_%s.txt" % (Path.DATA_FOLDER, s), 'r')
        outs = open("%s/data/new_outs_%s.txt" % (Path.DATA_FOLDER, s), 'r')
        i_content = [int(x.strip()) for x in index.readlines()]
        o_content = [int(x.strip()) for x in outs.readlines()]
        prev_i = 0
        for i in i_content:
            total += 1
            max_o = np.argmax(o_content[prev_i:prev_i + i])
            sorted_i = np.argsort([-x[0] for x in predictions[prev_i:prev_i + i]])
            if max_o in sorted_i[:1]:
                top1 += 1
            if max_o in sorted_i[:2]:
                top2 += 1
            if max_o in sorted_i[:5]:
                top5 += 1
            if max_o in sorted_i[:7]:
                top7 += 1
            if max_o in sorted_i[:10]:
                top10 += 1
            prev_i += i
        print("Top 1: %s" % (top1 * 100.0 / total))
        print("Top 2: %s" % (top2 * 100.0 / total))
        print("Top 5: %s" % (top5 * 100.0 / total))
        print("Top 7: %s" % (top7 * 100.0 / total))
        print("Top 10: %s" % (top7 * 100.0 / total))


if __name__ == "__main__":
    parameters = load_parameters()
    logging.info('Running training.')
    test_data(parameters)
    logging.info('Done!')
