from foodmenurecognition.variables.paths import Path


def load_parameters():

    # Input data params
    DATA_ROOT_PATH = Path.DATA_FOLDER
    DATA_WORD2VEC = Path.DATA_FOLDER + '/data/mini.h5'

    # preprocessed features
    DATASET_NAME = 'YELP-MVA'

    PRE_TRAINED_VOCABULARY_NAME = None

    IMAGES_LIST_FILES = {'train': 'data/links_train.txt',
                         'val': 'data/links_val.txt',
                         'test': 'data/links_test.txt',
                         }
    CNN_FILES = {'train': 'data/cnn_train.txt',
                    'val': 'data/cnn_val.txt',
                    'test': 'data/cnn_test.txt',
                    }
    DISHES_FILES = {'train': 'data/dishes_train.txt',
                    'val': 'data/dishes_val.txt',
                    'test': 'data/dishes_test.txt',
                    }

    OUT_FILES = {'train': 'data/outs_train.txt',
                 'val': 'data/outs_val.txt',
                 'test': 'data/outs_test.txt',
                 }

    # Dataset parameters
    INPUTS_IDS_DATASET = ['image', 'dish', 'cnn']  # Corresponding inputs of the dataset
    OUTPUTS_IDS_DATASET = ['distance']  # Corresponding outputs of the dataset
    INPUTS_IDS_MODEL = INPUTS_IDS_DATASET  # Corresponding inputs of the built model
    OUTPUTS_IDS_MODEL = OUTPUTS_IDS_DATASET  # Corresponding outputs of the built model

    # Word representation params
    TOKENIZATION_METHOD = 'tokenize_icann'  # Select which tokenization we'll apply:
    #  tokenize_basic, tokenize_aggressive, tokenize_soft,
    #  tokenize_icann or tokenize_questions

    FILL = 'end'  # whether we fill the 'end' or the 'start' of the sentence with 0s

    # Input image parameters
    IMG_FEAT_SIZE = 211  # Size of the image features
    CNN_SIZE = 1536

    # Output text parameters
    OUTPUT_VOCABULARY_SIZE = 0  # Size of the input vocabulary. Set to 0 for using all,
    # otherwise it will be truncated to these most frequent words.
    MAX_OUTPUT_TEXT_LEN = 30  # Maximum length of the output sequence
    # set to 0 if we want to use the whole answer as a single class
    MAX_OUTPUT_TEXT_LEN_TEST = 50  # Maximum length of the output sequence during test time
    MIN_OCCURRENCES_VOCAB = 0  # Minimum number of occurrences allowed for the words in the vocabulay.

    OPTIMIZER = 'Adadelta'  # Optimizer
    LR = 1.  # Learning rate. Recommended values - Adam 0.001 - Adadelta 1.0
    CLIP_C = 10.  # During training, clip gradients to this norm

    # Input text parameters
    TARGET_TEXT_EMBEDDING_SIZE = 300  # Source language word embedding size (ABiViRNet 301)

    IMAGE_TEXT_MAPPING = 300

    # Results plot and models storing parameters
    MODEL_NAME = DATASET_NAME

    MODEL_TYPE = 'food_img_embedding'

    STORE_PATH = 'trained_models/' + MODEL_NAME + '/'  # Models and evaluation results will be stored here
    DATASET_STORE_PATH = 'datasets/'  # Data-set instance will be stored here

    VERBOSE = 1  # Vqerbosity level
    RELOAD = 0  # If 0 start training from scratch, otherwise the model
    # Saved on epoch 'RELOAD' will be used
    REBUILD_DATASET = True  # Build again or use stored instance

    # ============================================
    parameters = locals().copy()
    return parameters
