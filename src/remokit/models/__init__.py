
"""Model utilities."""

from keras.models import Model


def submodel(model, last_layer):
    """Get a submodel."""
    last_layer = model.get_layer(last_layer)
    return Model(inputs=model.input, outputs=last_layer.output)
