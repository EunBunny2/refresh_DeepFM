
ALL_FIELDS = ['mlsfc','mcate_cd', 'Sex', 'Age', 'Month', 'time', 'day', 'fav_plc']
CONT_FIELDS = ['Age']
CAT_FIELDS = list(set(ALL_FIELDS).difference(CONT_FIELDS))

# Hyper-parameters for Experiment
NUM_BIN = 120 # 연속형 변수를 구간별로 나눌 때 구간 기준
BATCH_SIZE = 256
EMBEDDING_SIZE = 3