# Passion4ever

from functools import wraps

import sklearn.metrics as mets


def new_name(name):

    def decorator(old_func):

        @wraps(old_func)
        def new_func(*args, **kwargs):
            return old_func(*args, **kwargs)

        new_func.__name__ = name
        return new_func

    return decorator


Acc = new_name('accuracy')(mets.accuracy_score)

Sn = new_name('Sn')(mets.recall_score)

Mcc = new_name('mcc')(mets.matthews_corrcoef)

def Sp(y_true, y_pred):
    cm = mets.confusion_matrix(y_true, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    specificity = TN / (TN + FP)
    return specificity
