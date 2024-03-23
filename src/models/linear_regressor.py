import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class OrdinalFlexibleClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_layers=None):
        """
        Initializes a flexible classifier for ordinal regression.

        Parameters:
        - input_size: The size of the input features.
        - hidden_layers: None for a linear regressor, or a list of integers specifying the size of each hidden layer for an MLP.
        """
        super(OrdinalFlexibleClassifier, self).__init__()

        if hidden_layers is None:
            self.layers = nn.Sequential(nn.Linear(input_size, 1))
        else:
            layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]
            for i in range(1, len(hidden_layers)):
                layers.extend([nn.Linear(hidden_layers[i-1], hidden_layers[i]), nn.ReLU()])
            layers.append(nn.Linear(hidden_layers[-1], 1))
            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the classifier.

        Parameters:
        - x: Input tensor of shape (batch_size, input_size)

        Returns:
        - Scalar value representing the predicted change percentage.
        """
        return self.layers(x).squeeze(1)

class RegressorWithEncoder(pl.LightningModule):
    def __init__(self, encoder, input_shape, hidden_layers=None, learning_rate=1e-3):
        super().__init__()
        self.encoder = encoder  # The preloaded encoder, e.g., a ViT model

        # Infer the output size of the encoder
        self.encoder.eval()  # Ensure the encoder is in evaluation mode
        dummy_input = torch.randn(input_shape)  # Adjust the shape based on encoder's expected input
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)
        output_size = dummy_output.shape[1]

        self.regression_head = OrdinalFlexibleClassifier(input_size=output_size, hidden_layers=hidden_layers)
        self.learning_rate = learning_rate

        # Optionally, freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        embeddings = self.encoder(x)  # Get embeddings from encoder
        return self.regression_head(embeddings)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()  # Ensure y is a float tensor for MSE loss calculation
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.regression_head.parameters(), lr=self.learning_rate)
        return optimizer
