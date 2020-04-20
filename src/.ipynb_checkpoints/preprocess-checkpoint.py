import librosa
import numpy as np
import copy


def subset(X,y,ratio=0.33):
    """
    random subset of X,y
    Args:
        X (np.array): array containing the audio file as np.array
        y(np.array): targets of X
        ratio(float): relative size of the random subset
    Returns:
        a random subset of X,y
    """
    assert(X.shape[0] == y.shape[0])
    N = X.shape[0]
    indx = np.arange(N)
    np.random.shuffle(indx)
    indx = indx[:int(N*ratio)]
    return copy.copy(X[indx]),copy.copy(y[indx])

def add_shift(X,shift_max = 13000):
    """
    add a random shift to the elements in X
    Args:
        X (np.array): array containing the audio file as np.array
    Returns:
        X_shift(np.array): a randomly shifted version of X
    """
    shifts = np.random.randint(-shift_max,shift_max,X.shape[0])
    X_shift = np.zeros_like(X)
    for i,(x_smple,shift) in enumerate(zip(X,shifts)):
        x_smple = np.roll(x_smple, shift)
        if shift > 0:
            x_smple[:shift] = 0
        else:
            x_smple[shift:] = 0
        X_shift[i] = x_smple
    return X_shift      

def add_noise(X,noise_factor=3500):
    """
    add a random noise to the elements in X
    Args:
        X (np.array): array containing the audio file as np.array
        noise_factor(int): amplitude of the gaussian noise
    Returns:
        X(np.array): a gaussian noised version of X
    """
    noise = np.random.randn(*X.shape)
    X = X + noise_factor * noise
    return X

def add_pitch(X,sr=16000,n_steps=0.6):
    """
    add a pitch to the elements in X
    Args:
        X (np.array): array containing the audio file as np.array
    Returns:
        X(np.array): a gaussian noised version of X
    """
    return np.array([librosa.effects.pitch_shift(x_sample, sr, n_steps=n_steps) for x_sample in X])

def data_augment(X,y,rndom_noise=False,shift=False,pitch=False,ratio=0.4):
    """
    apply some data augmentation on the elements of X
    Args:
        X (np.array): array containing the audio file as np.array
        y(np.array(int)): targets of X
        rndom_noise(Bool): applies the random noise addition
        shift(Bool): applies the random shift addition
        pitch(Bool): applies the pitch tuning
    Returns:
        X(np.array): a gaussian noised version of X
    """
    X_ret = copy.deepcopy(X)
    y_ret = copy.deepcopy(y)
    if rndom_noise:
        X_noise,y_noise = subset(X,y,ratio)
        X_noise = add_noise(X_noise)
        X_ret = np.append(X_ret,X_noise,0)
        y_ret = np.append(y_ret,y_noise,0)
    if shift:
        X_shift,y_shift = subset(X,y,ratio)
        X_shift = add_shift(X_shift)
        X_ret = np.append(X_ret,X_shift,0)
        y_ret = np.append(y_ret,y_shift,0)
    if pitch:
        X_pitch,y_pitch = subset(X,y,ratio)
        X_pitch = add_pitch(X_pitch)
        X_ret = np.append(X_ret,X_pitch,0)
        y_ret = np.append(y_ret,y_pitch,0)
    return X_ret,y_ret   