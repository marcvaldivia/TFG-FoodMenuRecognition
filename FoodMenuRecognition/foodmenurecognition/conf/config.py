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
    DISHES_FILES = {'train': 'data/dishes_train.txt',
                    'val': 'data/dishes_val.txt',
                    'test': 'data/dishes_test.txt',
                    }

    OUT_FILES = {'train': 'data/outs_train.txt',
                 'val': 'data/outs_val.txt',
                 'test': 'data/outs_test.txt',
                 }

    # Dataset parameters
    INPUTS_IDS_DATASET = ['image', 'dish']  # Corresponding inputs of the dataset
    OUTPUTS_IDS_DATASET = ['corr']  # Corresponding outputs of the dataset
    INPUTS_IDS_MODEL = INPUTS_IDS_DATASET  # Corresponding inputs of the built model
    OUTPUTS_IDS_MODEL = OUTPUTS_IDS_DATASET  # Corresponding outputs of the built model

    # Evaluation params
    METRICS = [
        'multiclass_metrics']  # Metric used for evaluating model after each epoch (leave empty if only prediction is required)
    EVAL_ON_SETS = ['val', 'test']  # Possible values: 'train', 'val' and 'test' (external evaluator)
    EVAL_ON_SETS_KERAS = []  # Possible values: 'train', 'val' and 'test' (Keras' evaluator)
    START_EVAL_ON_EPOCH = 1  # First epoch where the model will be evaluated
    EVAL_EACH_EPOCHS = False  # Select whether evaluate between N epochs or N updates
    EVAL_EACH = 50  # Sets the evaluation frequency (epochs or updates)

    # Word representation params
    TOKENIZATION_METHOD = 'tokenize_icann'  # Select which tokenization we'll apply:
    #  tokenize_basic, tokenize_aggressive, tokenize_soft,
    #  tokenize_icann or tokenize_questions

    FILL = 'end'  # whether we fill the 'end' or the 'start' of the sentence with 0s
    TRG_LAN = 'en'  # Language of the outputs (mainly used for the Meteor evaluator)
    PAD_ON_BATCH = True  # Whether we take as many timesteps as the longes sequence of the batch
    # or a fixed size (MAX_OUTPUT_TEXT_LEN)

    # Input image parameters
    DATA_AUGMENTATION = False  # Apply data augmentation on input data (noise on features)
    DATA_AUGMENTATION_TYPE = ['random_selection']  # 'random_selection', 'noise'
    IMG_FEAT_SIZE = 211  # Size of the image features

    # Output text parameters
    OUTPUT_VOCABULARY_SIZE = 0  # Size of the input vocabulary. Set to 0 for using all,
    # otherwise it will be truncated to these most frequent words.
    MAX_OUTPUT_TEXT_LEN = 30  # Maximum length of the output sequence
    # set to 0 if we want to use the whole answer as a single class
    MAX_OUTPUT_TEXT_LEN_TEST = 50  # Maximum length of the output sequence during test time
    MIN_OCCURRENCES_VOCAB = 0  # Minimum number of occurrences allowed for the words in the vocabulay.

    # Optimizer parameters (see model.compile() function)
    LOSS = 'binary_crossentropy'

    OPTIMIZER = 'Adadelta'  # Optimizer
    LR = 1.  # Learning rate. Recommended values - Adam 0.001 - Adadelta 1.0
    CLIP_C = 10.  # During training, clip gradients to this norm

    # Training parameters
    MAX_EPOCH = 200  # Stop when computed this number of epochs
    BATCH_SIZE = 64  # ABiViRNet trained with BATCH_SIZE = 64

    PARALLEL_LOADERS = 1  # Parallel data batch loaders
    EPOCHS_FOR_SAVE = 1 if EVAL_EACH_EPOCHS else None  # Number of epochs between model saves (None for disabling epoch save)

    WRITE_VALID_SAMPLES = True  # Write valid samples in file
    SAVE_EACH_EVALUATION = True if not EVAL_EACH_EPOCHS else False  # Save each time we evaluate the model

    # Early stop parameters
    EARLY_STOP = True  # Turns on/off the early stop protocol
    PATIENCE = 20  # We'll stop if the val STOP_METRIC does not improve after this
    # number of evaluations

    STOP_METRIC = 'Bleu_4'  # Metric for the stop

    # Input text parameters
    TARGET_TEXT_EMBEDDING_SIZE = 300  # Source language word embedding size (ABiViRNet 301)

    IMAGE_TEXT_MAPPING = 300

    TRG_PRETRAINED_VECTORS = ""  # Path to pretrained vectors. (e.g. DATA_ROOT_PATH + '/DATA/word2vec.%s.npy' % TRG_LAN)
    # Set to None if you don't want to use pretrained vectors.
    # When using pretrained word embeddings, the size of the pretrained word embeddings must match with the word embeddings size.
    TRG_PRETRAINED_VECTORS_TRAINABLE = True  # Finetune or not the target word embedding vectors.

    # Regularizers
    WEIGHT_DECAY = 1e-4  # L2 regularization
    RECURRENT_WEIGHT_DECAY = 0.  # L2 regularization in recurrent layers

    USE_DROPOUT = True  # Use dropout
    DROPOUT_P = 0.2  # Percentage of units to drop

    USE_BATCH_NORMALIZATION = False  # If True it is recommended to deactivate Dropout
    BATCH_NORMALIZATION_MODE = 1  # See documentation in Keras' BN

    USE_PRELU = False  # use PReLU activations as regularizer
    USE_L2 = False  # L2 normalization on the features

    # Results plot and models storing parameters
    MODEL_NAME = DATASET_NAME

    MODEL_TYPE = 'Food_Img_Embedding'

    STORE_PATH = 'trained_models/' + MODEL_NAME + '/'  # Models and evaluation results will be stored here
    DATASET_STORE_PATH = 'datasets/'  # Dataset instance will be stored here

    VERBOSE = 1  # Vqerbosity level
    RELOAD = 0  # If 0 start training from scratch, otherwise the model
    # Saved on epoch 'RELOAD' will be used
    REBUILD_DATASET = True  # Build again or use stored instance
    MODE = 'training'  # 'training' or 'sampling' (if 'sampling' then RELOAD must
    # be greater than 0 and EVAL_ON_SETS will be used)
    RELOAD_PATH = None

    # Extra parameters for special trainings
    FORCE_RELOAD_VOCABULARY = False  # force building a new vocabulary from the training samples applicable if RELOAD > 1

    # CALLBACKS
    CLASSIFICATION_TYPE = 'single-label'
    OUTPUTS_TYPES = 'categorical'
    NUM_CLASSES = 2

    # ============================================
    parameters = locals().copy()
    return parameters
