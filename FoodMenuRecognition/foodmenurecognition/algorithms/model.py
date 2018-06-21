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


cache_dict = dict()


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


def train_model(params, epochs, distance=euclidean_similarity, loss="binary_crossentropy", cnn=0, sample_weight=True):

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


def get_results(params, epochs, distance, loss, cnn, sample_weight):
    num_it = 4
    key = "%s-%s-%s-%s" % (distance, loss, cnn, sample_weight)
    if key in cache_dict.keys():
        return cache_dict[key]
    sum_values = [[list(), list(), list(), list()] for _ in range(epochs)]
    for _ in range(num_it):
        train_model(params, epochs, distance=distance, loss=loss, cnn=cnn,
                    sample_weight=sample_weight)
        for epoch in range(1, epochs + 1):
            acc, r_loss = test_model(params, 'val', epoch)
            acc_test, r_loss_test = test_model(params, 'test', epoch)
            sum_values[epoch - 1][0].append(acc)
            sum_values[epoch - 1][1].append(r_loss)
            sum_values[epoch - 1][2].append(acc_test)
            sum_values[epoch - 1][3].append(r_loss_test)
    best_epoch, best_r_loss, best_r_loss_test = 0, 1.0, 1.0
    historical = list()
    for epoch in range(epochs):
        acc, r_loss = np.median(sum_values[epoch][0]), np.median(sum_values[epoch][1])
        acc_test, r_loss_test = np.median(sum_values[epoch][2]), np.median(sum_values[epoch][3])
        historical.append((acc, r_loss, acc_test, r_loss_test))
        if r_loss < best_r_loss:
            best_epoch, best_r_loss, best_r_loss_test = epoch, r_loss, r_loss_test
    cache_dict[key] = (best_epoch, best_r_loss, best_r_loss_test, historical)
    return cache_dict[key]


def eval_results(params, epochs):
    best_epoch, best_r_loss = 0, 1.0
    historical = list()
    for epoch in range(1, epochs + 1):
        acc, r_loss = test_model(params, 'test', epoch)
        historical.append((acc, r_loss))
        if r_loss < best_r_loss:
            best_epoch, best_r_loss = epoch, r_loss
    return best_epoch, best_r_loss, historical


def add_to_df(df, level, best, r_loss, epoch, historical):
    df.loc["level: %s / type: %s best" % (level, best), 'best'] = best
    df.loc["level: %s / type: %s best" % (level, best), 'r_loss'] = r_loss
    df.loc["level: %s / type: %s best" % (level, best), 'epoch'] = epoch
    df.loc["level: %s / type: %s best" % (level, best), 'historical'] = str(historical)


def new_grid_search(params, epochs, split_kind):
    df = pd.DataFrame()
    file_name = "Grid_Search_%s.csv" % split_kind
    best_measure, best_loss = (euclidean_similarity, 1.0, 1, list()), ('binary_crossentropy', 1.0, 1, list())
    best_cnn, best_sample = (0, 1.0, 1, list()), (False, 1.0, 1, list())
    try:
        for distance in [(pearson_similarity, "pearson"), (euclidean_similarity, "euclidean")]:
            best_epoch, best_r_loss, best_r_loss_test, historical = \
                get_results(params, epochs, distance=distance[0], loss=best_loss[0],
                            cnn=best_cnn[0], sample_weight=best_sample[0])
            add_to_df(df, 'measure', distance[0], best_r_loss_test, best_epoch, historical)
            df.to_csv(file_name)
            if best_measure[1] > best_r_loss_test:
                best_measure = (distance[0], best_r_loss_test, best_epoch, historical)
        for loss in ["binary_crossentropy", contrastive_loss]:
            best_epoch, best_r_loss, best_r_loss_test, historical = \
                get_results(params, epochs, distance=best_measure[0], loss=loss,
                            cnn=best_cnn[0], sample_weight=best_sample[0])
            add_to_df(df, 'loss', loss, best_r_loss_test, best_epoch, historical)
            df.to_csv(file_name)
            if best_loss[1] > best_r_loss_test:
                best_loss = (loss, best_r_loss_test, best_epoch, historical)
        for cnn in [0, 1, 2]:
            best_epoch, best_r_loss, best_r_loss_test, historical = \
                get_results(params, epochs, distance=best_measure[0], loss=best_loss[0],
                            cnn=cnn, sample_weight=best_sample[0])
            add_to_df(df, 'cnn', cnn, best_r_loss_test, best_epoch, historical)
            df.to_csv(file_name)
            if best_cnn[1] > best_r_loss_test:
                best_cnn = (cnn, best_r_loss_test, best_epoch, historical)
        for sample_weight in [True, False]:
            best_epoch, best_r_loss, best_r_loss_test, historical = \
                get_results(params, epochs, distance=best_measure[0], loss=best_loss[0],
                            cnn=best_cnn[0], sample_weight=sample_weight)
            add_to_df(df, 'sample_weight', sample_weight, best_r_loss_test, best_epoch, historical)
            df.to_csv(file_name)
            if best_sample[1] > best_r_loss_test:
                best_sample = (sample_weight, best_r_loss_test, best_epoch, historical)
        df.to_csv(file_name)
    except Exception as ex:
        logging.error(ex)


if __name__ == "__main__":
    parameters = load_parameters()
    get_results(parameters, 5, euclidean_similarity, "binary_crossentropy", 1, False)
