from multiprocessing import Value
from logging import getLogger
import torch

_GLOBAL_SEED = 0
logger = getLogger()

class MaskCollator(object):
    def __init__(
        self,
        ratio=(0.4, 0.6),
        input_length=224,  # Assuming input_length is the total number of patches in 1D
        patch_size=16,  # Assuming patch_size still relevant for 1D sequence length
    ):
        super(MaskCollator, self).__init__()
        self.patch_size = patch_size
        self.length = input_length // patch_size  # Sequence length in patches for 1D
        self.ratio = ratio
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        with self._itr_counter.get_lock():
            self._itr_counter.value += 1
            return self._itr_counter.value

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating items into a batch for 1D input
        # 1. sample block size using seed and specified ratio
        # 2. return enc mask and pred mask
        '''
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        ratio = self.ratio[0] + torch.rand(1, generator=g).item() * (ratio[1] - self.ratio[0])
        num_patches = self.length
        num_keep = int(num_patches * (1. - ratio))

        collated_masks_pred, collated_masks_enc = [], []
        for _ in range(B):
            m = torch.randperm(num_patches)
            collated_masks_enc.append([m[:num_keep]])
            collated_masks_pred.append([m[num_keep:]])

        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred
