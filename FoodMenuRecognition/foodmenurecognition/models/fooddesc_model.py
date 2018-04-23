import logging

import pandas as pd
from keras.layers import *
from keras.models import model_from_json, Model
from keras_wrapper.cnn_model import Model_Wrapper

from foodmenurecognition.utils.text_uri import standardized_uri


class FoodDesc_Model(Model_Wrapper):

    def resumeTrainNet(self, ds, params, out_name=None):
        pass

    def __init__(self, params, type='Food_Img_Embedding', verbose=1, structure_path=None, weights_path=None,
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
        # if params['TRG_PRETRAINED_VECTORS'] is not None:
        if True:
            if self.verbose > 0:
                logging.info("<<< Loading pretrained word vectors from: " + params['TRG_PRETRAINED_VECTORS'] + " >>>")
            self.trg_word_vectors = None  # np.load(os.path.join(params['TRG_PRETRAINED_VECTORS'])).item()
            data = pd.read_hdf("/Users/yoda/git/TFG-FoodMenuRecognition/mini.h5")
            self.trg_embedding_weights = np.random.rand(params['INPUT_VOCABULARY_SIZE'],
                                                        params['TARGET_TEXT_EMBEDDING_SIZE'])
            for word, index in self.vocabularies[self.ids_inputs[1]]['words2idx'].iteritems():
                try:
                    self.trg_embedding_weights[index, :] = data.loc[standardized_uri("en", word)] \
                        .values.reshape(1, 300)
                except:
                    pass
            self.trg_embedding_weights = [self.trg_embedding_weights]
            self.trg_embedding_weights_trainable = params['TRG_PRETRAINED_VECTORS_TRAINABLE']
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

    def setOptimizer(self, **kwargs):
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

    def Food_Img_Embedding(self, params):

        image = Input(name=self.ids_inputs[0], shape=tuple([params['IMG_FEAT_SIZE']]))

        emb_image = Dense(params['IMAGE_TEXT_MAPPING'])(image)

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

        def t(x):
            A, B = x
            fsp = x[0] - K.mean(x[0], axis=-1, keepdims=True)
            fst = x[1] - K.mean(x[1], axis=-1, keepdims=True)
            sum_up = K.sum(fsp * fst, axis=-1, keepdims=True)
            mul_down = K.sqrt(K.sum(K.square(fsp), axis=-1, keepdims=True)) * K.sqrt(K.sum(K.square(fst), axis=-1,
                                                                                           keepdims=True))
            div = sum_up / mul_down
            return K.abs(div)
            # return K.l2_normalize(K.sum(K.abs(A - B), axis=1, keepdims=True))

        def cosine_distance(vests):
            x, y = vests
            x = K.l2_normalize(x, axis=-1)
            y = K.l2_normalize(y, axis=-1)
            return -K.mean(x * y, axis=-1, keepdims=True)

        dist = Lambda(t, name=self.ids_outputs[0])([emb_food, emb_image])

        self.model = Model(input=[image, food_word], output=dist)
