import numpy as np

PAD = 0
DISEASE_MIN, DISEASE_MAX = 22, 1277
MED_MIN, MED_MAX = 1278, 1284
DEATH = 1288
VOCAB_SIZE = 1289
N_MED = 7
FIRST, CONTINUATION, RESTART = 0, 1, 2
ACTION_NAMES = ('FIRST_RECORDED', 'CONTINUATION', 'RESTART')
DTYPE = np.dtype([(k, '<u4') for k in ('ID', 'AGE', 'EVENT', 'DOSE', 'DUR')])

def medication(token):
    return (token >= MED_MIN) & (token <= MED_MAX)

def disease(token):
    return (token >= DISEASE_MIN) & (token <= DISEASE_MAX)

def clinical(token):
    return disease(token) | medication(token) | (token == DEATH)

def supported_tokens():
    return list(range(DISEASE_MIN, MED_MAX + 1)) + [DEATH]