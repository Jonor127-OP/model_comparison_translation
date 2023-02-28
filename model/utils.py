import torch

def get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id=0):

    batch_size = trg_token_ids_batch.shape[0]

    # Same as src_mask but we additionally want to mask tokens from looking forward into the future tokens
    # Note: wherever the mask value is true we want to attend to that token, otherwise we mask (ignore) it.
    tgt_mask = trg_token_ids_batch != 0
    tgt_mask = tgt_mask[:, None, None, :]
    trg_padding_mask = tgt_mask.repeat(1, 1, tgt_mask.shape[-1], 1)
    sequence_length = trg_token_ids_batch.shape[1]  # trg_token_ids shape = (B, T) where T max trg token-sequence length
    trg_no_look_forward_mask = torch.triu(torch.ones((1, 1, sequence_length, sequence_length)) == 1).transpose(2, 3)

    # logic AND operation (both padding mask and no-look-forward must be true to attend to a certain target token)
    trg_mask = trg_padding_mask & trg_no_look_forward_mask  # final shape = (B, 1, T, T)

    return trg_mask