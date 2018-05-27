import logging
from timeit import default_timer as timer

import pandas as pd
from keras.layers import *
from keras_wrapper.cnn_model import loadModel
from sklearn.metrics import label_ranking_loss

from foodmenurecognition.algorithms.dataset import DataSet
from foodmenurecognition.conf.config import load_parameters
from foodmenurecognition.models.fooddesc_model import FoodDesc_Model
from foodmenurecognition.utils.prepare_data import build_dataset, build_dataset_test, _build_dataset_test
from foodmenurecognition.variables.paths import Path

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def contrastive_loss(y_true, y_pred):
    '''
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def exponent_neg_manhattan_distance(vects):
    left, right = vects
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return 1 - K.exp(-K.sum(K.abs(left - right), axis=-1, keepdims=True))


def pearson_sim(x):
    fsp = x[0] - K.mean(x[0], axis=-1, keepdims=True)
    fst = x[1] - K.mean(x[1], axis=-1, keepdims=True)
    sum_up = K.sum(fsp * fst, axis=-1, keepdims=True)
    mul_down = K.sqrt(K.sum(K.square(fsp), axis=-1, keepdims=True)) * K.sqrt(K.sum(K.square(fst), axis=-1,
                                                                                   keepdims=True))
    div = sum_up / mul_down
    return K.abs(div)


def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return K.abs(K.abs(-K.mean(x * y, axis=-1, keepdims=True)))


def eu_distance(vests):
    y_true, y_pred = vests
    return K.sqrt(K.maximum(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True), K.epsilon()))


def train_model(params, epochs, distance=eu_distance, loss="binary_crossentropy", cnn=True):
    # Load data
    dataset = build_dataset(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][1]]

    params['distance'] = distance
    params['cnn'] = cnn
    params['LOSS'] = loss

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
    params['MAX_EPOCH'] += params['RELOAD']

    # Update optimizer either if we are loading or building a model
    food_model.params = params
    food_model.setOptimizer()

    total_start_time = timer()

    logger.debug('Starting training!')
    training_params = {
        'normalize': False,
        'n_epochs': epochs,
        'batch_size': 32,
        'n_parallel_loaders': 1,
        'metric_check': 'accuracy'
    }

    food_model.trainNet(dataset, training_params)

    total_end_time = timer()
    time_difference = total_end_time - total_start_time
    logging.info('In total is {0:.2f}s = {1:.2f}m'.format(time_difference, time_difference / 60.0))


def test_model(params, s, i):
    build_dataset(params)
    food_model = loadModel(params['STORE_PATH'], i)
    food_model.setOptimizer()

    build_dataset_test(params, 'val')
    build_dataset_test(params, 'test')
    dataset = _build_dataset_test(params)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][1]]  # Load model
    # Apply model predictions
    params_prediction = {
        'predict_on_sets': [s],
        'normalize': False,
        'n_parallel_loaders': 1,
        'verbose': True
    }
    predictions = food_model.predictNet(dataset, params_prediction)[s]

    acc, r_loss = 0, list()
    total, top1, top2, top5, top7, top10 = 0, 0, 0, 0, 0, 0
    index = open("%s/data/index_%s.txt" % (Path.DATA_FOLDER, s), 'r')
    outs = open("%s/data/new_outs_%s.txt" % (Path.DATA_FOLDER, s), 'r')
    i_content = [int(x.strip()) for x in index.readlines()]
    o_content = [int(x.strip()) for x in outs.readlines()]
    prev_i = 0
    for i in i_content:
        total += 1
        r_loss.append(label_ranking_loss([o_content[prev_i:prev_i + i]],
                                         [[-x[0] for x in predictions[prev_i:prev_i + i].tolist()]]))
        max_o = np.argmax(o_content[prev_i:prev_i + i])
        sorted_i = np.argsort([x[0] for x in predictions[prev_i:prev_i + i]])
        acc += (i - list(sorted_i).index(max_o)) * 1.0 / i
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
    print("Top 10: %s" % (top10 * 100.0 / total))
    print("Acc: %s" % (acc / total))
    print("Ranking Loss: %s" % np.mean(r_loss))
    print("Total: %s" % total)
    return (top1 * 100.0 / total), (top2 * 100.0 / total), (top5 * 100.0 / total), (top7 * 100.0 / total), (
            top10 * 100.0 / total), (acc / total), np.mean(r_loss)


def grid_search(params, epochs):
    df = pd.DataFrame()
    try:
        for ing in [True, False]:
            for d_type in [0, 1, 2]:
                d = DataSet(Path.DATA_FOLDER, split_kind=d_type, ingredients=ing)
                d.execute()
                d.execute_files('train')
                d.execute_files('val')
                d.execute_files('test')
                params['IMG_FEAT_SIZE'] = 211 if not ing else 1270
                for cnn in [True, False]:
                    for distance in [(pearson_sim, "pearson"), (eu_distance, "euclidean"), (cosine_distance, "cosine")]:
                        train_model(params, distance[0], epochs, cnn)
                        for s in ['val', 'test']:
                            for i in range(1, epochs + 1):
                                top1, top2, top5, top7, top10, acc, r_loss = test_model(params, s, i)
                                df.loc['epoch: %s / set: %s / distance: %s / cnn: %s / type: %s / ing.: %s'
                                       % (i, s, distance[1], cnn, d_type, ing), 'top1'] = top1
                                df.loc['epoch: %s / set: %s / distance: %s / cnn: %s / type: %s / ing.: %s'
                                       % (i, s, distance[1], cnn, d_type, ing), 'top2'] = top2
                                df.loc['epoch: %s / set: %s / distance: %s / cnn: %s / type: %s / ing.: %s'
                                       % (i, s, distance[1], cnn, d_type, ing), 'top5'] = top5
                                df.loc['epoch: %s / set: %s / distance: %s / cnn: %s / type: %s / ing.: %s'
                                       % (i, s, distance[1], cnn, d_type, ing), 'top7'] = top7
                                df.loc['epoch: %s / set: %s / distance: %s / cnn: %s / type: %s / ing.: %s'
                                       % (i, s, distance[1], cnn, d_type, ing), 'top10'] = top10
                                df.loc['epoch: %s / set: %s / distance: %s / cnn: %s / type: %s / ing.: %s'
                                       % (i, s, distance[1], cnn, d_type, ing), 'acc'] = acc
                                df.loc['epoch: %s / set: %s / distance: %s / cnn: %s / type: %s / ing.: %s'
                                       % (i, s, distance[1], cnn, d_type, ing), 'r_loss'] = r_loss
    except Exception as ex:
        logging.error(ex)
    df.to_csv("GridSearch.csv")


def eval_results(params, epochs):
    best_epoch, best_r_loss = 0, 1.0
    historical = list()
    for epoch in range(1, epochs + 1):
        top1, top2, top5, top7, top10, acc, r_loss = test_model(params, 'val', epoch)
        historical.append((top1, top2, top5, top7, top10, acc, r_loss))
        if r_loss < best_r_loss:
            best_epoch, best_r_loss = epoch, r_loss
    return best_epoch, best_r_loss, historical


def create_ds(split_kind, ingredients):
    s = DataSet(Path.DATA_FOLDER, split_kind=split_kind, ingredients=ingredients)
    s.execute()
    s.execute_division()
    s.execute_files('train')
    s.execute_files('val')
    s.execute_files('test')


def add_to_df(df, level, best, r_loss, epoch, historical):
    df.loc['level: %s' % level, 'best'] = best
    df.loc['level: %s' % level, 'r_loss'] = r_loss
    df.loc['level: %s' % level, 'epoch'] = epoch
    df.loc['level: %s' % level, 'historical'] = str(historical)


def new_grid_search(params, epochs, split_kind):
    df = pd.DataFrame()
    best_measure, best_loss = (pearson_sim, 1.0, 1, list()), ('binary_crossentropy', 1.0, 1, list())
    best_ing, best_cnn = (False, 1.0, 1, list()), (False, 1.0, 1, list())
    try:
        create_ds(split_kind, best_ing[0])
        for distance in [(pearson_sim, "pearson"), (eu_distance, "euclidean"),
                         (exponent_neg_manhattan_distance, "manhattan"), (cosine_distance, "cosine")]:
            train_model(params, epochs, distance=distance[0], loss=best_loss[0], cnn=best_cnn[0])
            best_epoch, best_r_loss, historical = eval_results(params, epochs)
            if best_measure[1] > best_r_loss:
                best_measure = (distance[0], best_r_loss, best_epoch, historical)
        add_to_df(df, 'measure', best_measure[0], best_measure[1], best_measure[2], best_measure[3])
        df.to_csv("GridSearch.csv")
        for loss in ["binary_crossentropy", "mean_squared_error", contrastive_loss]:
            train_model(params, epochs, distance=best_measure[0], loss=loss, cnn=best_cnn[0])
            best_epoch, best_r_loss, historical = eval_results(params, epochs)
            if best_loss[1] > best_r_loss:
                best_loss = (loss, best_r_loss, best_epoch, historical)
        add_to_df(df, 'loss', best_loss[0], best_loss[1], best_loss[2], best_loss[3])
        df.to_csv("GridSearch.csv")
        for cnn in [True, False]:
            train_model(params, epochs, distance=best_measure[0], loss=best_loss[0], cnn=cnn)
            best_epoch, best_r_loss, historical = eval_results(params, epochs)
            if best_cnn[1] > best_r_loss:
                best_cnn = (cnn, best_r_loss, best_epoch)
        add_to_df(df, 'cnn', best_cnn[0], best_cnn[1], best_cnn[2], best_cnn[3])
        df.to_csv("GridSearch.csv")
        for ing in [True, False]:
            params['IMG_FEAT_SIZE'] = 211 if not ing else 1310
            create_ds(split_kind=split_kind, ingredients=ing)
            train_model(params, epochs, distance=best_measure[0], loss=best_loss[0], cnn=best_cnn[0])
            best_epoch, best_r_loss, historical = eval_results(params, epochs)
            if best_ing[1] > best_r_loss:
                best_ing = (ing, best_r_loss, best_epoch)
        add_to_df(df, 'ingredients', best_ing[0], best_ing[1], best_ing[2], best_ing[3])
        df.to_csv("GridSearch.csv")
    except Exception as ex:
        logging.error(ex)


if __name__ == "__main__":
    parameters = load_parameters()
    # train_model(parameters, epochs=5, distance=eu_distance, cnn=False)
    # test_model(parameters, 'val', 5)
    new_grid_search(parameters, 10, 2)
