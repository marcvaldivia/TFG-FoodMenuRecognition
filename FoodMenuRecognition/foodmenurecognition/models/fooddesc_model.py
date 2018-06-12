import logging

import pandas as pd
from keras.layers import *
from keras.models import model_from_json, Model
from keras_wrapper.cnn_model import Model_Wrapper

from foodmenurecognition.utils.text_uri import standardized_uri


class FoodDescModel(Model_Wrapper):

    def resume_train_net(self, ds, params, out_name=None):
        pass

    def __init__(self, params, type='food_img_embedding', verbose=1, structure_path=None, weights_path=None,
                 model_name=None, vocabularies=None, store_path=None, set_optimizer=True, clear_dirs=True):
        super(self.__class__, self).__init__(type=type, model_name=model_name,
                                             silence=verbose == 0, models_path=store_path, inheritance=True)

        self.__toprint = ['_model_type', 'name', 'model_path', 'verbose']

        self.verbose = verbose
        self._model_type = type
        self.params = params
        self.vocabularies = vocabularies
        self.ids_inputs = params['INPUTS_IDS_MODEL']
        self.ids_outputs = params['OUTPUTS_IDS_MODEL']
        # Sets the model name and prepares the folders for storing the models
        self.setName(model_name, models_path=store_path, clear_dirs=clear_dirs)

        # Prepare target word embedding
        if params['DATA_WORD2VEC'] is not None:
            if self.verbose > 0:
                logging.info("<<< Loading pretrained word vectors from: " + params['DATA_WORD2VEC'] + " >>>")
            self.trg_word_vectors = None  # np.load(os.path.join(params['TRG_PRETRAINED_VECTORS'])).item()
            data = pd.read_hdf(params['DATA_WORD2VEC'])
            self.trg_embedding_weights = np.random.rand(params['INPUT_VOCABULARY_SIZE'],
                                                        params['TARGET_TEXT_EMBEDDING_SIZE'])
            for word, index in self.vocabularies[self.ids_inputs[1]]['words2idx'].iteritems():
                try:
                    self.trg_embedding_weights[index, :] = data.loc[standardized_uri("en", word)] \
                        .values.reshape(1, 300)
                except Exception as _:
                    try:
                        self.trg_embedding_weights[index, :] = data.loc[standardized_uri("es", word)] \
                            .values.reshape(1, 300)
                    except Exception as _:
                        pass
            self.trg_embedding_weights = [self.trg_embedding_weights]
            self.trg_embedding_weights_trainable = True
            del self.trg_word_vectors
            del data
        else:
            self.trg_embedding_weights = np.random.rand(params['INPUT_VOCABULARY_SIZE'],
                                                        params['TARGET_TEXT_EMBEDDING_SIZE'])
            self.trg_embedding_weights = [self.trg_embedding_weights]
            self.trg_embedding_weights_trainable = True

        # Prepare model
        if structure_path:
            # Load a .json model
            if self.verbose > 0:
                logging.info("<<< Loading model structure from file " + structure_path + " >>>")
            self.model = model_from_json(open(structure_path).read())
        else:
            # Build model from scratch
            if hasattr(self, type):
                if self.verbose > 0:
                    logging.info("<<< Building '" + type + "' Video Captioning Model >>>")
                eval('self.' + type + '(params)')
            else:
                raise Exception('Video_Captioning_Model type "' + type + '" is not implemented.')

        # Load weights from file
        if weights_path:
            if self.verbose > 0:
                logging.info("<<< Loading weights from file " + weights_path + " >>>")
            self.model.load_weights(weights_path)

        # Print information of self
        if verbose > 0:
            print str(self)
            self.model.summary()
        if set_optimizer:
            self.setOptimizer()

    def setOptimizer(self):
        super(self.__class__, self).setOptimizer(lr=self.params['LR'],
                                                 clipnorm=self.params['CLIP_C'],
                                                 loss=self.params['LOSS'],
                                                 optimizer=self.params['OPTIMIZER'],
                                                 sample_weight_mode='temporal' if self.params.get('SAMPLE_WEIGHTS',
                                                                                                  False) else None)

    def __str__(self):
        obj_str = '-----------------------------------------------------------------------------------\n'
        class_name = self.__class__.__name__
        obj_str += '\t\t' + class_name + ' instance\n'
        obj_str += '-----------------------------------------------------------------------------------\n'

        # Print pickled attributes
        for att in self.__toprint:
            obj_str += att + ': ' + str(self.__dict__[att])
            obj_str += '\n'

        obj_str += '\n'
        obj_str += 'MODEL params:\n'
        obj_str += str(self.params)
        obj_str += '\n'
        obj_str += '-----------------------------------------------------------------------------------'

        return obj_str

    # ------------------------------------------------------- #
    #       DEFINED MODELS
    # ------------------------------------------------------- #

    def food_img_embedding(self, params):

        # l1 = Dense(params['IMAGE_TEXT_MAPPING'], activation='relu')
        # l2 = Dropout(0.1)
        # l3 = Dense(params['IMAGE_TEXT_MAPPING'], activation='relu')
        # l4 = Dropout(0.1)
        # l5 = Dense(params['IMAGE_TEXT_MAPPING'], activation='relu')

        image_i = Input(name=self.ids_inputs[0], shape=tuple([params['IMG_FEAT_SIZE']]))
        image = Dense(params['IMAGE_TEXT_MAPPING'])(image_i)
        # image = l1(image)
        # image = l2(image)
        # image = l3(image)
        # image = l4(image)
        # image = l5(image)

        cnn_i = Input(name=self.ids_inputs[2], shape=tuple([params['CNN_SIZE']]))
        cnn = Dense(params['IMAGE_TEXT_MAPPING'], activation="sigmoid")(cnn_i)
        # cnn = l1(cnn)
        # cnn = l2(cnn)
        # cnn = l3(cnn)
        # cnn = l4(cnn)
        # cnn = l5(cnn)

        food_word = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        shared_emb = Embedding(params['INPUT_VOCABULARY_SIZE'],
                               params['TARGET_TEXT_EMBEDDING_SIZE'],
                               name='target_word_embedding',
                               weights=self.trg_embedding_weights,
                               trainable=self.trg_embedding_weights_trainable,
                               mask_zero=True)
        emb = shared_emb(food_word)
        emb_food = LSTM(params['IMAGE_TEXT_MAPPING'],
                        return_sequences=False,
                        name='encoder_LSTM')(emb)
        # emb_food = Dense(params['IMAGE_TEXT_MAPPING'])(emb_food)
        # emb_food = l1(emb_food)
        # emb_food = l2(emb_food)
        # emb_food = l3(emb_food)
        # emb_food = l4(emb_food)
        # emb_food = l5(emb_food)

        added = Add()([image, cnn])

        dist = Lambda(params['distance'], name=self.ids_outputs[0])([added, emb_food]) if params['cnn'] == 1 \
            else Lambda(params['distance'], name=self.ids_outputs[0])([image, emb_food]) if params['cnn'] == 0 \
            else Lambda(params['distance'], name=self.ids_outputs[0])([cnn, emb_food])

        self.model = Model(input=[image_i, cnn_i, food_word], output=dist)
