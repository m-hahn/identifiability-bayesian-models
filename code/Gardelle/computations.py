import math

def setData(**kwargs):
    global STIMULUS_SPACE_VOLUME
    STIMULUS_SPACE_VOLUME = kwargs["STIMULUS_SPACE_VOLUME"]
    global GRID
    GRID = kwargs["GRID"]




def computeResources(volume, inverse_variance):
    assert (volume.sum() - 2*math.pi) < 0.01, "the computation is presupposing that the sensory space has volume 2\pi, so that the inverse_variance corresponds to the kappa parameter"
    return volume * math.sqrt(inverse_variance) * GRID / STIMULUS_SPACE_VOLUME
def computeResourcesWithStimulusNoise(volume, inverse_variance, inverse_variance_stimulus):
    """
     inverse_variance is expected to be the inverse sensory noise variance (sensory noise precision)
     inverse_variance_stimulus is expected to be the inverse stimulus noise variance (stimulus noise precision)
    """
    return 1/(1/computeResources(volume, inverse_variance).pow(2) + 1/inverse_variance_stimulus).sqrt()

