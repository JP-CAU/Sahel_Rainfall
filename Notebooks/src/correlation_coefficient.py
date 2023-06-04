from tensorflow.keras import backend as K

def correlation_coefficient(y_true, y_pred):
    
    """
    Customized metric function to compute the correlation coefficient between model predictions and true targets.
    
    Parameters:
    ===========
    y_true: Tensorflow tensor with true targets.
    y_pred: Tensorflow tensor with model predictions.
    
    Returns:
    ========
    Pearson correlation coefficient.
    
    """
    
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(xm * ym)
    r_den = K.sqrt(K.sum(K.square(xm))) * K.sqrt(K.sum(K.square(ym)))
    r = r_num / r_den
    return r