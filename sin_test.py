import os
import argparse
#from datasets.candles_regression import make_timeseries
from src.datasets.timeseries import make_timeseries
#from src.datasets.candles_coin_classification import make_timeseries
#from src.datasets.candles_classification import make_timeseries
from src.transforms import make_transforms
from src.models.linear_regressor import RegressorWithEncoder
from src.models.linear_classifier import ClassifierWithEncoder
from src.helper import load_checkpoint, init_model
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt

class ActivateEncoderGradCallback(Callback):
    def __init__(self, activate_epoch):
        super().__init__()
        self.activate_epoch = activate_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == self.activate_epoch:
            print(f"Activating encoder weight updates at epoch: {trainer.current_epoch + 1}")
            pl_module.set_encoder_grad(True)
            pl_module.learning_rate = 1e-5

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')


def load_model(args, device, checkpoint_path):

    patch_size = args['mask']['patch_size']  # patch-size for model training
    crop_size = args['data']['crop_size']
    num_channels = args['data']['num_channels']
    model_name = args['meta']['model_name']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']

    # Initialize model
    encoder, _ = init_model(
        device=device,
        patch_size=patch_size,
        num_channels=num_channels,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
    )
    encoder.to(device)

    # Load checkpoint
    encoder, _, target_encoder, _, _, _ = load_checkpoint(
        device=device,
        r_path=checkpoint_path,
        encoder=encoder,
        predictor=None,  # Assuming not needed for inference
        target_encoder=encoder,  # Assuming not needed for inference
        opt=None,  # Assuming not needed for inference
        scaler=None)  # Assuming not needed for inference

    # Lock weights
    # for param in encoder.parameters():
    #     param.requires_grad = False

    return target_encoder

def main(args):
    devices = args.devices
    # -- load script params

    with open(args.fname, 'r') as y_file:
        args = yaml.load(y_file, Loader=yaml.FullLoader)
        print('loaded params...')

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    r_file = args['meta']['read_checkpoint']
  
    # --
    batch_size = 32# args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    crop_size = args['data']['crop_size']
    # --



    transform = make_transforms()

    # -- init data-loaders/samplers
    dataset, loader, sampler = make_timeseries(
            transform=transform,
            batch_size=batch_size,
            sequence_length=crop_size,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=1,
            rank=0,
            drop_last=True)
    
    # for batch_data, frequencies in loader:
    #     print(f"Batch Shape: {batch_data.shape}, Frequencies: {frequencies}")
    #     plt.figure(figsize=(15, 10))  # Set the figure size for better visibility

    #     # Assuming batch_data is shaped [batch_size, num_channels, sequence_length]
    #     # and you want to plot all sequences in the batch
    #     for i in range(batch_data.size(0)):
    #         plt.subplot(5, 7, i+1)  # Change subplot grid dimensions as needed based on batch size
    #         for channel in range(batch_data.size(1)):
    #             plt.plot(batch_data[i, channel].numpy(), label=f'Channel {channel}')
    #         plt.title(f'Frequency: {frequencies[i].item():.2f}')
    #         plt.tight_layout()
    #         plt.legend()

    #     plt.show()



    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')

    load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    encoder = load_model(args, devices[0], load_path)
    input_shape = dataset[0][0].shape
    # add an additional dimension for the batch size
    input_shape = (1,) + input_shape
    print(input_shape)
    model = RegressorWithEncoder(encoder=encoder, input_shape=input_shape, hidden_layers=None, learning_rate=1e-3)
    model.to(devices[0])

    # Initialize a trainer
    activate_encoder_grad_callback = ActivateEncoderGradCallback(activate_epoch=190)
    trainer = pl.Trainer(max_epochs=200,callbacks=[activate_encoder_grad_callback])
    print(trainer.callbacks)
    # Train the model
    trainer.fit(model, loader)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)