import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy

class FlexibleClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_layers=None, dropout_rate=0.3):
        super(FlexibleClassifier, self).__init__()
        layers = []
        current_input_size = input_size# Adjusted input size

        if hidden_layers is None:
            layers.append(nn.Linear(current_input_size, num_classes))
        else:
            for i, hidden_layer_size in enumerate(hidden_layers):
                layers.append(nn.Linear(current_input_size, hidden_layer_size))
                if dropout_rate > 0:  # Add dropout if the rate is greater than 0
                    layers.append(nn.Dropout(dropout_rate))
                layers.append(nn.ReLU())
                current_input_size = hidden_layer_size  # Update input size for the next layer
            
            layers.append(nn.Linear(hidden_layers[-1], num_classes))  # Output layer

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
class ClassifierWithEncoder(pl.LightningModule):
    def __init__(self, encoder, input_shape, num_classes, hidden_layers=None, learning_rate=1e-3):
        super().__init__()
        self.encoder = encoder

        self.encoder.eval()
        device = next(encoder.parameters()).device
        dummy_input = torch.randn(input_shape, device=device)
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)
        output_size = dummy_output.shape[-1]

        self.set_encoder_grad(False)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classification_head = FlexibleClassifier(input_size=output_size, num_classes=num_classes, hidden_layers=hidden_layers)
        self.learning_rate = learning_rate

    def forward(self, x):

        # Process embeddings with the encoder
        embeddings = self.encoder(x)
        embeddings = embeddings.transpose(1, 2)
        pooled_embeddings = self.avg_pool(embeddings).squeeze(-1)

        output = self.classification_head(pooled_embeddings)
        return output
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        # Calculate loss
        loss = F.cross_entropy(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Calculate accuracy
        pred_classes = torch.argmax(y_pred, dim=1)
        correct_count = (pred_classes == y).float().sum()
        accuracy = correct_count / y.shape[0]
        
        #print(y_pred.shape)
        # Log accuracy
        self.log("train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)  # Forward pass on validation batch
        loss = F.cross_entropy(y_pred, y)  # Compute validation loss
        # Calculate accuracy
        pred_classes = torch.argmax(y_pred, dim=1)
        correct_count = (pred_classes == y).float().sum()
        accuracy = correct_count / y.shape[0]
        # Log accuracy
        print(pred_classes)
        print(y)
        self.log("val_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_acc": accuracy}

    # def on_validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
    #     self.log("val_loss", avg_loss, prog_bar=True)
    #     self.log("val_acc", avg_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.classification_head.parameters(), lr=self.learning_rate)
        return optimizer

    def set_encoder_grad(self, requires_grad):
        for param in self.encoder.parameters():
            param.requires_grad = requires_grad