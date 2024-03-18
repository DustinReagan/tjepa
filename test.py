import argparse
from src.datasets.timeseries import make_timeseries
from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.transforms import make_transforms
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')

def main(args, resume_preempt=False):
    print(args)
    print(args.fname)
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
    load_model = args['meta']['load_checkpoint'] or resume_preempt
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

    mask_collator = MBMaskCollator(
        input_length=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    transform = make_transforms()

    # -- init data-loaders/samplers
    _, unsupervised_loader, unsupervised_sampler = make_timeseries(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=1,
            rank=0,
            drop_last=True)
    print(len(unsupervised_loader))

    for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):
        print('itr:', itr)
        print('udata shape:', udata.shape)
        print(len(masks_enc))
        print(len(masks_pred))
        break


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)