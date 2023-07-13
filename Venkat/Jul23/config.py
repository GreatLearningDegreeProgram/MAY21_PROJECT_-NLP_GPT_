import re

BANNER_IMAGE1 ='images/caronavirus banner.jpg'
SIDEBAR_IMAGE ='images/Sidebar2.jpg'
COLOR_CODE = ['rgb(102, 0, 51)', 'rgb(204, 0, 102)', 'rgb(255, 51, 153)', 'rgb(102, 255, 255)', 'rgb(204, 204, 255)']
BERT_MAX_TOKEN = 512
GPT2_MAX_TOKEN = 1024
GPT2_MAX_LEN = 100
TOP_K_READER = 5
TOP_K_RETRIEVER = 10
USE_GPU = True

MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)
DAMPING  = 0.85  # damping coefficient, usually is .85
MIN_DIFF = 1e-5  # convergence threshold
STEPS = 100  # iteration steps

TEXT_STR = None
SENTENCES = None
PR_VECTOR = None

EMBEDDING_DIM = 512
QRY_EMBEDDING_MODEL = "sentence-transformers/clip-ViT-B-32"
QRY_TYPE = "text"
DOC_EMBEDDING_MODELS = "sentence-transformers/clip-ViT-B-32"

HAYSTACK_READER = 'TRANSFORMER'

NER_COLORS = {
    "CC": 1, "CD": 2, "DT": 3, "EX": 4, "FW": 5, "IN": 6, "JJ": 7, "JJR": 8, "JJS": 9, "MD": 10, "NN": 11, "NNP": 12, 
    "NNPS": 13, "NNS": 14, "O": 0, "PDT": 15, "POS": 16, "PRP": 17, "RB": 18, "RBR": 19, "RBS": 20, "RP": 21, "SYM": 22,
    "TO": 23, "UH": 24, "VB": 25, "VBD": 26, "VBG": 27, "VBN": 28, "VBP": 29, "VBZ": 30, "WDT": 31, "WP": 32, "WRB": 33
}
CLASSIFIER_LABEL = ['Analytical','Compare','Interpretative','Experimental','Survey']

doc_dir = 'text_file'
content_image ='images\content'
MULTIMODAL_IMG_DIR = 'images/content'
MULTIMODAL_IMAGE_WIDTH= 250