import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class FlexibleRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_layers=None):
        super(FlexibleRegressor, self).__init__()
        if hidden_layers is None:
            self.layers = nn.Sequential(
                nn.Linear(input_size, 1),
                #nn.Tanh()  # Add Tanh activation for the output layer
            ) 
        else:
            layers = [nn.Linear(input_size, hidden_layers[0])]
            print(f"Initializing first layer with input size {input_size} and output size {hidden_layers[0]}")
            for i in range(1, len(hidden_layers)):
                layers.extend([
                    nn.Linear(hidden_layers[i-1], hidden_layers[i]),
                    nn.ReLU()  # Assuming ReLU activations for hidden layers
                ])
                print(f"Initializing layer {i} with input size {hidden_layers[i-1]} and output size {hidden_layers[i]}")
            layers.extend([
                nn.Linear(hidden_layers[-1], 1),
                #nn.Tanh()  # Add Tanh activation for the output layer
            ])
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
        # Ensure the dummy input is on the same device as the encoder
        device = next(encoder.parameters()).device
        dummy_input = torch.randn(input_shape, device=device)  # Adjust the shape based on encoder's expected input
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)
        output_size = dummy_output.shape[-1]
        
        print("Dummy input shape:", dummy_input.shape)
        print("Dummy output shape:", dummy_output.shape)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.regression_head = FlexibleRegressor(input_size=output_size, hidden_layers=hidden_layers)
        self.learning_rate = learning_rate

        # Optionally, freeze the encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        embeddings = self.encoder(x)
        # Ensure embeddings is not None and correctly shaped
        
        # encoder's output shape is [batch_size, seq_len, embedding_dim]
        # and we want to average across the seq_len dimension.
        # Transpose to [batch_size, embedding_dim, seq_len] for AdaptiveAvgPool1d
        embeddings = embeddings.transpose(1, 2)

        # Apply average pooling
        pooled_embeddings = self.avg_pool(embeddings).squeeze(-1)  # Removing the last dimension

        
        
        # Pass through the regression head
        output = self.regression_head(pooled_embeddings)

        # pass the last sequence through the regression head
        #output = self.regression_head(embeddings[:, -1, :])

        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_mean = y.float().mean()
        y_std = y.float().std()
        y_normalized = (y.float() - y_mean) / (y_std + 1e-6)  # Add a small epsilon to avoid division by zero

        y_pred = self(x)

        # Depending on your downstream tasks or evaluation metrics,
        # you might want to denormalize predictions or keep them normalized.
        # Here, we proceed with normalized predictions for calculating MSE loss.

        loss = F.mse_loss(y_pred, y_normalized)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.regression_head.parameters(), lr=self.learning_rate)
        return optimizer
