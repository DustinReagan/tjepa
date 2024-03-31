import math
from multiprocessing import Value
from logging import getLogger
import torch

# Global seed for reproducibility.
_GLOBAL_SEED = 0
# Initialize a logger for logging information and warnings.
logger = getLogger()

class MaskCollator(object):
    def __init__(
        self,
        input_length=224,  # The length of input data.
        patch_size=16,  # The size of each patch in the input data.
        enc_mask_scale=(0.2, 0.8),  # Scale range for encoder masks.
        pred_mask_scale=(0.2, 0.8),  # Scale range for predictor masks.
        nenc=1,  # Number of encoder masks to generate.
        npred=2,  # Number of predictor masks to generate.
        min_keep=4,  # Minimum number of patches to keep unmasked.
        allow_overlap=False  # Whether to allow overlap between encoder and predictor masks.
    ):
        super(MaskCollator, self).__init__()
        self.patch_size = patch_size
        self.length = input_length // patch_size  # Calculate the length in terms of patches.
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        # Initialize a counter for iterations, shared across worker processes.
        self._itr_counter = Value('i', -1)
        # Log the initialization parameters.
        logger.info(f'input_length: {input_length} patch_size: {patch_size} length: {self.length} enc_mask_scale: {enc_mask_scale} pred_mask_scale: {pred_mask_scale} nenc: {nenc} npred: {npred} min_keep: {min_keep} allow_overlap: {allow_overlap}')

    def step(self):
        # Increment the iteration counter and return the new value.
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale):
        # Randomly sample a block scale within the given scale range.
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.length * mask_scale)

        # Calculate the block length based on the mask scale.
        l = int(round(math.sqrt(max_keep)))#int(round(max_keep / 2))#int(round(math.sqrt(max_keep)))
        while l >= self.length:
            l -= 1

        return l

    def _sample_block_mask(self, length, acceptable_regions=None):
        
        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            start = torch.randint(0, self.length - 1, (1,))
            mask = torch.zeros((self.length), dtype=torch.int32)
            mask[start:start+length] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) >= self.min_keep
            if not valid_mask:
                
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.length), dtype=torch.int32)
        mask_complement[start:start+length] = 0
        # --
        #print('length: ', length, mask)
        return mask, mask_complement

    def __call__(self, batch):
        '''
        This method is called when the MaskCollator object is used to process a batch of data.
        It generates masks for the encoder and predictor based on the initialized parameters.
        '''
        B = len(batch)  # Batch size.

        # Default collate function to process the batch data.
        collated_batch = torch.utils.data.default_collate(batch)

        # Seed for random number generation to ensure reproducibility.
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        # Sample block sizes for predictor and encoder masks.
        p_size = self._sample_block_size(generator=g, scale=self.pred_mask_scale)
        #print(f'p_size: {p_size}')
        e_size = self._sample_block_size(generator=g, scale=self.enc_mask_scale)
        #print(  f'e_size: {e_size}')

        #print(f'p_size: {p_size} e_size: {e_size}')
        #logger.info(f'p_size: {p_size} e_size: {e_size}')

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.length
        min_keep_enc = self.length
        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                # Sample predictor masks.
                mask, mask_C = self._sample_block_mask(p_size)
                # if mask is 0d tensor, then expand it to 1d
                if len(mask.shape) == 0:
                    mask = mask.unsqueeze(0)
                masks_p.append(mask)
                masks_C.append(mask_C)
                #print('predictor mask: ', mask)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions = None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                # Sample encoder masks with consideration of acceptable regions.
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                # if mask is 0d tensor, then expand it to 1d
                if len(mask.shape) == 0:
                    mask = mask.unsqueeze(0)
                masks_e.append(mask)
                #print('encoder mask: ', mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        # Adjust the masks to the minimum keep size for both predictor and encoder.
        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred
