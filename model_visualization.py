from torchview import draw_graph
from ril_prediction_functions import *
from torchinfo import summary


model = SpectralCNN()
graph = draw_graph(model, input_size= (32, 1, 242))
graph.visual_graph.render("outputs/model_architecture", format="png")

summary(SpectralCNN(), input_size=(32, 1, 242))