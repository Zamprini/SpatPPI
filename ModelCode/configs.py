from yacs.config import CfgNode as CN

_C = CN()

# Protein feature extractor
_C.PROTEIN = CN()
_C.PROTEIN.EMBEDDING_DIM = [64, 64, 64]
_C.PROTEIN.NODE_DIM = 41
_C.PROTEIN.EDGE_DIM = 7

#
_C.ATTENTION = CN()
_C.ATTENTION.HEADS = 1
_C.ATTENTION.BETA = 0.2
_C.ATTENTION.NEIGHBORS = 20

# BCN setting
_C.BCC = CN()
_C.BCC.HEADS = 2

# MLP decoder
_C.DECODER = CN()
_C.DECODER.NAME = "MLP"
_C.DECODER.IN_DIM = 128
_C.DECODER.HIDDEN_DIM = 256 
_C.DECODER.OUT_DIM = 64
_C.DECODER.BINARY = 1

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 50
_C.SOLVER.BATCH_SIZE = 4 
_C.SOLVER.NUM_WORKERS = 0
_C.SOLVER.SEED = 2048

# RESULT
_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "./result"
_C.RESULT.SAVE_MODEL = True


def get_cfg_defaults():
    return _C.clone()