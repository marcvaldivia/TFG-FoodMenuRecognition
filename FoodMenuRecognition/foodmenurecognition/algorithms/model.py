import pandas as pd
from keras.layers import *
from sklearn.metrics import label_ranking_loss
import logging
import pickle

from keras_wrapper.cnn_model import loadModel

from foodmenurecognition.conf.config import load_parameters
from foodmenurecognition.models.fooddesc_model import FoodDescModel
from foodmenurecognition.utils.prepare_data import build_dataset, build_dataset_val_test
from foodmenurecognition.variables.paths import Path


def contrastive_loss(y_true, y_pred):
    '''
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return 1 - K.mean(y_true * K.square(y_pred) +
                      (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def exponent_neg_manhattan_distance(vects):
    left, right = vects
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left - right), axis=-1, keepdims=True))


def pearson_similarity(x):
    fsp = x[0] - K.mean(x[0], axis=-1, keepdims=True)
    fst = x[1] - K.mean(x[1], axis=-1, keepdims=True)
    sum_up = K.sum(fsp * fst, axis=-1, keepdims=True)
    mul_down = K.sqrt(K.sum(K.square(fsp), axis=-1, keepdims=True)) * K.sqrt(K.sum(K.square(fst), axis=-1,
                                                                                   keepdims=True))
    div = sum_up / mul_down
    return K.abs(div)


def euclidean_similarity(vests):
    y_true, y_pred = vests
    return 1 / (1 + K.sqrt(K.maximum(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True), K.epsilon())))


def train_model(params, epochs, distance=euclidean_similarity, loss="binary_crossentropy", cnn=1, sample_weight=False):

    # Load data
    params['sample_weight'] = sample_weight
    dataset = build_dataset(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][1]]

    params['distance'] = distance
    params['cnn'] = cnn
    params['LOSS'] = loss

    # Build model
    food_model = FoodDescModel(params,
                               type=params['MODEL_TYPE'],
                               verbose=params['VERBOSE'],
                               model_name=params['MODEL_NAME'] + '_reloaded',
                               vocabularies=dataset.vocabulary,
                               store_path=params['STORE_PATH'],
                               set_optimizer=False,
                               clear_dirs=False)

    # Define the inputs and outputs mapping from our Dataset instance to our model
    input_mapping = dict()
    for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
        if len(food_model.ids_inputs) > i:
            pos_source = dataset.ids_inputs.index(id_in)
            id_dest = food_model.ids_inputs[i]
            input_mapping[id_dest] = pos_source
    food_model.setInputsMapping(input_mapping)

    output_mapping = dict()
    for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
        if len(food_model.ids_outputs) > i:
            pos_target = dataset.ids_outputs.index(id_out)
            id_dest = food_model.ids_outputs[i]
            output_mapping[id_dest] = pos_target
    food_model.setOutputsMapping(output_mapping)

    food_model.setOptimizer()

    # Update optimizer either if we are loading or building a model
    food_model.params = params
    food_model.setOptimizer()

    training_params = {
        'normalize': False,
        'n_epochs': epochs,
        'batch_size': 64,
        'n_parallel_loaders': 1
    }

    food_model.trainNet(dataset, training_params)


def test_model(params, s, i):
    food_model = loadModel(params['STORE_PATH'], i)
    food_model.setOptimizer()

    dataset = build_dataset_val_test(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][1]]

    params_prediction = {
        'predict_on_sets': [s],
        'normalize': False,
        'n_parallel_loaders': 1,
        'verbose': True
    }
    predictions = food_model.predictNet(dataset, params_prediction)[s]

    with open("%s/data/predictions_%s.txt" % (Path.DATA_FOLDER, s), "wb") as fp:
        pickle.dump(predictions, fp)

    total, acc, r_loss = 0, 0, list()
    index = open("%s/data/index_%s.txt" % (Path.DATA_FOLDER, s), 'r')
    outs = open("%s/data/new_outs_%s.txt" % (Path.DATA_FOLDER, s), 'r')
    i_content = [int(x.strip()) for x in index.readlines()]
    o_content = [int(x.strip()) for x in outs.readlines()]
    prev_i = 0
    for i in i_content:
        total += 1
        r_loss.append(label_ranking_loss([o_content[prev_i:prev_i + i]],
                                         [[x[0] for x in predictions[prev_i:prev_i + i].tolist()]]))
        max_o = np.argmax(o_content[prev_i:prev_i + i])
        sorted_i = np.argsort([-x[0] for x in predictions[prev_i:prev_i + i]])
        acc += (i - list(sorted_i).index(max_o)) * 1.0 / i
        prev_i += i
    print("Acc: %s" % (acc / total))
    print("Ranking Loss: %s" % np.mean(r_loss))
    print("Total: %s" % total)
    return (acc / total), np.mean(r_loss)


def eval_results(params, epochs):
    best_epoch, best_r_loss = 0, 1.0
    historical = list()
    for epoch in range(1, epochs + 1):
        acc, r_loss = test_model(params, 'test', epoch)
        historical.append((acc, r_loss))
        if r_loss < best_r_loss:
            best_epoch, best_r_loss = epoch, r_loss
    return best_epoch, best_r_loss, historical


if __name__ == "__main__":
    parameters = load_parameters()
    train_model(parameters, 1)
    test_model(parameters, 'test', 1)
