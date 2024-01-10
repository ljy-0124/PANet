import torch.nn as nn


class MetricModel(nn.Module):
    def __init__(self, base_model, projection_in_dim, projection_out_dim):
        super(MetricModel, self).__init__()
        # Load the base model
        self.base_model = base_model
        # Replace the last layer with the custom projection layer
        projection_layer = nn.Linear(projection_in_dim, projection_out_dim)
        self.base_model.head.fc = projection_layer

    def forward(self, x):
        features = self.base_model(x)
        embeddings = features.view(features.size(0), -1)
        return embeddings

class transform_MetricModel(nn.Module):
    def __init__(self, base_model, projection_in_dim, projection_out_dim):
        super(transform_MetricModel, self).__init__()

        # Load the base model
        self.model = base_model
        # Replace the last layer with the custom projection layer
        projection_layer = nn.Linear(projection_in_dim, projection_out_dim)
        self.model.head.layers.head = projection_layer

    def forward(self, x):
        features = self.model(x)
      #  print("features.shape:",features.shape)
        embeddings = features.view(features.size(0), -1)
        return embeddings
