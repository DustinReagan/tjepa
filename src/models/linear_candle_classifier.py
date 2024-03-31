import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics.functional import accuracy

class FlexibleClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_layers=None, dropout_rate=0.2):
        super(FlexibleClassifier, self).__init__()
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate

        if hidden_layers is None:
            self.mlp = nn.Sequential(
                nn.Linear(input_size, num_classes)
            )
        else:
            mlp_layers = [nn.Linear(14 * 5, hidden_layers[0]), nn.ReLU(), nn.Dropout(dropout_rate)]
            for i in range(1, len(hidden_layers)):
                mlp_layers.extend([
                    nn.Linear(hidden_layers[i-1], hidden_layers[i]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
            #mlp_layers.append(nn.Linear(hidden_layers[-1], num_classes))
            self.mlp = nn.Sequential(*mlp_layers)

        # This layer will combine MLP output and pooled_embeddings
        self.combine_and_classify = nn.Linear(hidden_layers[-1] + input_size, num_classes) if hidden_layers else None

    def forward(self, x, embeddings=None):
        x = self.mlp(x)
        if embeddings is not None:
            x = torch.cat((x, embeddings), dim=1)
            x = self.combine_and_classify(x)
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
        # embedding shape is (batch_size, num_sequences, embedding_size)
        
        # get the last sequence of embeddings
        #embeddings = embeddings[:,:, -1]


        #print('embeddings shape: ', embeddings.shape)
        embeddings = self.avg_pool(embeddings).squeeze(-1)

        # Extract the last candle data correctly given the shape
        last_candle_data = x[:, :, -14:].reshape(x.shape[0], -1)  # Adjust as per your previous setup
        
        # Feed combined features into the classification head
        # Now 'pooled_embeddings' is passed as an additional input to 'classification_head'
        output = self.classification_head(last_candle_data, embeddings)
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