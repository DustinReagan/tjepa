import os
import argparse
from src.datasets.candles_classification import make_timeseries
from src.transforms import make_transforms
from src.models.linear_regressor import RegressorWithEncoder
from src.helper import load_checkpoint, init_model
import pytorch_lightning as pl
import yaml

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
    _, _, target_encoder, _, _, _ = load_checkpoint(
        device=device,
        r_path=checkpoint_path,
        encoder=encoder,
        predictor=None,  # Assuming not needed for inference
        target_encoder=encoder,  # Assuming not needed for inference
        opt=None,  # Assuming not needed for inference
        scaler=None)  # Assuming not needed for inference

    # Lock weights
    for param in encoder.parameters():
        param.requires_grad = False

    return target_encoder

def main(args, resume_preempt=False):
    print(args)
    print(args.fname)
    devices = args.devices
    # -- load script params

    with open(args.fname, 'r') as y_file:
        args = yaml.load(y_file, Loader=yaml.FullLoader)
        print('loaded params...')

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']

    # -- DATA


    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    crop_size = args['data']['crop_size']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks


    transform = make_transforms()

    # -- init data-loaders/samplers
    dataset, loader, sampler = make_timeseries(
            transform=transform,
            batch_size=batch_size,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=1,
            rank=0,
            drop_last=True)


    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')

    load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    encoder = load_model(args, devices[0], load_path)
    # Calculate input size dynamically or set it based on your data preprocessing
    input_shape = dataset[0][0].shape
    # add an additional dimension for the batch size
    input_shape = (1,) + input_shape
    print(input_shape)
    model = RegressorWithEncoder(encoder=encoder, input_shape=input_shape, hidden_layers=[512], learning_rate=1e-4)


    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=10)

    # Train the model
    trainer.fit(model, loader)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)