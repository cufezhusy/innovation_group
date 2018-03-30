from keras.utils import plot_model
from keras.models import load_model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

model = load_model("final_model.h5")
plot_model(model,to_file='model.png',show_shapes=True)