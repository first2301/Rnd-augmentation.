# from pycaret.classification import *

# def train_caret_clf(data, target):
#     s = setup(data=data, target=target, session_id=123, normalize = True, normalize_method = 'minmax')
#     models = compare_models(sort='Accuracy')
#     result = pull()
#     prepro_compare_df = result.iloc[:, 2:].T # DataFrame transpose
#     prepro_compare_df.columns = result['Model']
#     return prepro_compare_df

import ray
from pycaret.classification import ClassificationExperiment

@ray.remote
def train_caret_clf(data, target):
    s = ClassificationExperiment()
    # s.setup(data=data, target=target, session_id=123, normalize = True, normalize_method = 'minmax')
    s.setup(data=data, target=target, session_id=123)
    
    models_to_exclude = ['lightgbm']
    models = s.compare_models(exclude=models_to_exclude, sort='F1')

    # models = s.compare_models(sort='F1') 
    result = s.pull()
    prepro_compare_df = result.iloc[:, 1:].T 
    prepro_compare_df.columns = result['Model']
    return prepro_compare_df