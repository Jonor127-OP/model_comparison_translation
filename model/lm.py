
"""
    Contains the implementation of decoder-only transformer as known as language model like GPT-x.
    Certain modifications:
    1. LayerNorm (before instead of after)
    2. Dropout (Added additionally to attention weights and point-wise feed-forward net sublayer
"""


import math
import copy


import torch
import torch.nn as nn


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, model_dimension, number_of_heads, number_of_layers, dropout_probability, log_attention_weights=False):
        super().__init__()

        self.embedding = Embedding(vocab_size, model_dimension)
        self.pos_embedding = PositionalEncoding(model_dimension, dropout_probability)

        mha = MultiHeadedAttention(model_dimension, number_of_heads, dropout_probability, log_attention_weights)
        pwn = PositionwiseFeedForwardNet(model_dimension, dropout_probability)
        layer = DecoderLayer(model_dimension, dropout_probability, mha, pwn)

        self.decoder = Decoder(layer, number_of_layers)

        self.decoder_generator = DecoderGenerator(model_dimension, vocab_size)
        self.init_params()

    def init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, token_ids_batch, mask):
        embeddings_batch = self.embedding(token_ids_batch)
        embeddings_batch = self.pos_embedding(embeddings_batch)
        representations_batch = self.decoder(embeddings_batch, mask)
        log_probs = self.decoder_generator(representations_batch)

        return log_probs


    def generate_beam_search(self, input_token_ids, max_len, beam_size, device='cuda'):
        def get_top_k(sorted_probs, indices, k):
            top_k_probs, top_k_indices = [], []
            for i in range(k):
                top_k_probs.append(sorted_probs[i])
                top_k_indices.append(indices[i])
            return top_k_probs, top_k_indices

        if device == 'cuda':
            input_token_ids = input_token_ids.cuda()

        hypotheses = [(input_token_ids, 0)]

        for _ in range(max_len):
            new_hypotheses = []

            for hypothesis, score in hypotheses:
                if hypothesis[-1] == 1:
                    new_hypotheses.append((hypothesis, score))
                    continue

                input_mask = (hypothesis != 0).unsqueeze(1)
                input_embeddings = self.embedding(hypothesis)
                input_embeddings = self.pos_embedding(input_embeddings)

                input_representations = self.encoder(input_embeddings, input_mask)
                output_log_probs = self.decoder_generator(input_representations)

                predicted_log_probs = output_log_probs[:, -1, :]
                sorted_log_probs, sorted_indices = torch.sort(predicted_log_probs, descending=True)
                top_k_probs, top_k_indices = get_top_k(sorted_log_probs[0], sorted_indices[0], beam_size)

                for i in range(beam_size):
                    new_hypothesis = torch.cat((hypothesis, top_k_indices[i].unsqueeze(0).unsqueeze(0).cuda()), 1)
                    new_score = score + top_k_probs[i].item()
                    new_hypotheses.append((new_hypothesis, new_score))

            hypotheses = sorted(new_hypotheses, key=lambda x: x[1], reverse=True)[:beam_size]

        best_hypothesis = hypotheses[0][0]
        return best_hypothesis

    def generate_greedy(self, input_token_ids, max_len, cuda=False):
        if cuda:
            input_token_ids = input_token_ids.cuda()

        eos_reached = False

        while True:
            input_mask = self.get_masks_and_count_tokens_trg(input_token_ids, cuda=cuda)

            input_embeddings = self.embedding(input_token_ids)
            input_embeddings = self.pos_embedding(input_embeddings)

            input_representations = self.decoder(input_embeddings, input_mask)
            output_log_probs = self.decoder_generator(input_representations)

            predicted_log_probs = output_log_probs[:, -1, :]
            most_probable_token_indices = torch.argmax(predicted_log_probs, dim=-1).cpu().numpy()

            predicted_id = most_probable_token_indices[-1]

            if predicted_id == 1 or input_token_ids.shape[1] == max_len:
                eos_reached = True

            if eos_reached:
                break

            if cuda:
                input_token_ids = torch.cat((input_token_ids,
                                             torch.unsqueeze(torch.tensor(most_probable_token_indices[-1]).cuda(),
                                                             0).unsqueeze(0)), 1)
            else:
                input_token_ids = torch.cat(
                    (input_token_ids, torch.unsqueeze(torch.tensor(most_probable_token_indices[-1]), 0).unsqueeze(0)),
                    1)

        return input_token_ids


    def get_masks_and_count_tokens_trg(self, trg_token_ids_batch, pad_token_id=0, cuda=True):

        # Same as src_mask but we additionally want to mask tokens from looking forward into the future tokens
        # Note: wherever the mask value is true we want to attend to that token, otherwise we mask (ignore) it.
        tgt_mask = trg_token_ids_batch != 0
        tgt_mask = tgt_mask[:, None, None, :]
        trg_padding_mask = tgt_mask.repeat(1, 1, tgt_mask.shape[-1], 1)
        sequence_length = trg_token_ids_batch.shape[1]  # trg_token_ids shape = (B, T) where T max trg token-sequence length
        if cuda:
            trg_no_look_forward_mask = torch.triu(torch.ones((1, 1, sequence_length, sequence_length)) == 1).transpose(2, 3).cuda()
        else:
            trg_no_look_forward_mask = torch.triu(torch.ones((1, 1, sequence_length, sequence_length)) == 1).transpose(2, 3)

        # logic AND operation (both padding mask and no-look-forward must be true to attend to a certain target token)
        trg_mask = trg_padding_mask & trg_no_look_forward_mask  # final shape = (B, 1, T, T)

        return trg_mask


#
# Decoder architecture
#

class Decoder(nn.Module):

    def __init__(self, decoder_layer, number_of_layers):
        super().__init__()
        assert isinstance(decoder_layer, DecoderLayer), f'Expected DecoderLayer got {type(decoder_layer)}.'

        self.decoder_layers = get_clones(decoder_layer, number_of_layers)
        self.norm = nn.LayerNorm(decoder_layer.model_dimension)

    def forward(self, trg_embeddings_batch, src_representations_batch=None, trg_mask=None, src_mask=None):
        # Just update the naming so as to reflect the semantics of what this var will become
        trg_representations_batch = trg_embeddings_batch

        # Forward pass through the decoder stack
        for decoder_layer in self.decoder_layers:
            # Target mask masks pad tokens as well as future tokens (current target token can't look forward)
            if src_representations_batch is not None and trg_mask is not None and src_mask is not None:
                trg_representations_batch = decoder_layer(trg_representations_batch, src_representations_batch, trg_mask, src_mask)
            else:
                trg_representations_batch = decoder_layer(trg_representations_batch)

        # Not mentioned explicitly in the paper (a consequence of using LayerNorm before instead of after the sublayer
        # check out the SublayerLogic module)
        return self.norm(trg_representations_batch)


class DecoderLayer(nn.Module):

    def __init__(self, model_dimension, dropout_probability, multi_headed_attention, pointwise_net):
        super().__init__()
        num_of_sublayers_decoder = 3
        self.sublayers = get_clones(SublayerLogic(model_dimension, dropout_probability), num_of_sublayers_decoder)

        self.trg_multi_headed_attention = copy.deepcopy(multi_headed_attention)
        self.src_multi_headed_attention = copy.deepcopy(multi_headed_attention)
        self.pointwise_net = pointwise_net

        self.model_dimension = model_dimension

    def forward(self, trg_representations_batch, src_representations_batch=None, trg_mask=None, src_mask=None):
        # Define anonymous (lambda) function which only takes trg_representations_batch as input
        # The inputs which are not passed into lambdas are "cached" here that's why the thing works.
        decoder_trg_self_attention = lambda trb: self.trg_multi_headed_attention(query=trb, key=trb, value=trb, mask=trg_mask)

        trg_representations_batch = self.sublayers[0](trg_representations_batch, decoder_trg_self_attention)

        if src_representations_batch is not None and src_mask is not None:
            srb = src_representations_batch  # simple/short alias
            decoder_src_attention = lambda trb: self.src_multi_headed_attention(query=trb, key=srb, value=srb, mask=src_mask)
            trg_representations_batch = self.sublayers[1](trg_representations_batch, decoder_src_attention)

        trg_representations_batch = self.sublayers[2](trg_representations_batch, self.pointwise_net)

        return trg_representations_batch


#
# Helper modules (designed with modularity in mind) and organized top to bottom.
#


# Note: the original paper had LayerNorm AFTER the residual connection and addition operation
# multiple experiments I found showed that it's more effective to do it BEFORE, how did they figure out which one is
# better? Experiments! There is a similar thing in DCGAN and elsewhere.
class SublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, representations_batch, sublayer_module):
        # Residual connection between input and sublayer output, details: Page 7, Chapter 5.4 "Regularization",
        return representations_batch + self.dropout(sublayer_module(self.norm(representations_batch)))


class DecoderGenerator(nn.Module):
    def __init__(self, model_dimension, vocab_size):
        super().__init__()

        self.linear = nn.Linear(model_dimension, vocab_size)


    def forward(self, trg_representations_batch):
        # Project from D (model dimension) into V (target vocab size) and apply the log softmax along V dimension
        return self.linear(trg_representations_batch)


class PositionwiseFeedForwardNet(nn.Module):
    """
        It's position-wise because this feed forward net will be independently applied to every token's representation.
        Representations batch is of the shape (batch size, max token sequence length, model dimension).
        This net will basically be applied independently to every token's representation (you can think of it as if
        there was a nested for-loop going over the batch size and max token sequence length dimensions
        and applied this net to token representations. PyTorch does this auto-magically behind the scenes.
    """
    def __init__(self, model_dimension, dropout_probability, width_mult=4):
        super().__init__()

        self.linear1 = nn.Linear(model_dimension, width_mult * model_dimension)
        self.linear2 = nn.Linear(width_mult * model_dimension, model_dimension)

        # This dropout layer is not explicitly mentioned in the paper but it's common to use to avoid over-fitting
        self.dropout = nn.Dropout(p=dropout_probability)
        self.relu = nn.ReLU()

    def forward(self, representations_batch):
        return self.linear2(self.dropout(self.relu(self.linear1(representations_batch))))


class MultiHeadedAttention(nn.Module):
    """
        This module already exists in PyTorch. The reason I implemented it here from scratch is that
        PyTorch implementation is super complicated as they made it as generic/robust as possible whereas
        on the other hand I only want to support a limited use-case.
        Also this is arguable the most important architectural component in the Transformer model.
        Additional note:
        This is conceptually super easy stuff. It's just that matrix implementation makes things a bit less intuitive.
        If you take your time and go through the code and figure out all of the dimensions + write stuff down on paper
        you'll understand everything. Also do check out this amazing blog for conceptual understanding:
        https://jalammar.github.io/illustrated-transformer/
        Optimization notes:
        qkv_nets could be replaced by Parameter(torch.empty(3 * model_dimension, model_dimension)) and one more matrix
        for bias, which would make the implementation a bit more optimized. For the sake of easier understanding though,
        I'm doing it like this - using 3 "feed forward nets" (without activation/identity hence the quotation marks).
        Conceptually both implementations are the same.
        PyTorch's query/key/value are of different shape namely (max token sequence length, batch size, model dimension)
        whereas I'm using (batch size, max token sequence length, model dimension) because it's easier to understand
        and consistent with computer vision apps (batch dimension is always first followed by the number of channels (C)
        and image's spatial dimensions height (H) and width (W) -> (B, C, H, W).
        This has an important optimization implication, they can reshape their matrix into (B*NH, S/T, HD)
        (where B - batch size, S/T - max src/trg sequence length, NH - number of heads, HD - head dimension)
        in a single step and I can only get to (B, NH, S/T, HD) in single step
        (I could call contiguous() followed by view but that's expensive as it would incur additional matrix copy)
    """

    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)  # identity activation hence "nets"
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended, not explicitly mentioned in paper
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        self.log_attention_weights = log_attention_weights  # should we log attention weights
        self.attention_weights = None  # for visualization purposes, I cache the weights here (translation_script.py)

    def attention(self, query, key, value, mask):
        # Step 1: Scaled dot-product attention, Page 4, Chapter 3.2.1 "Scaled Dot-Product Attention"
        # Notation: B - batch size, S/T max src/trg token-sequence length, NH - number of heads, HD - head dimension
        # query/key/value shape = (B, NH, S/T, HD), scores shape = (B, NH, S, S), (B, NH, T, T) or (B, NH, T, S)
        # scores have different shapes as MHA is used in 3 contexts, self attention for src/trg and source attending MHA
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)

        # Step 2: Optionally mask tokens whose representations we want to ignore by setting a big negative number
        # to locations corresponding to those tokens (force softmax to output 0 probability on those locations).
        # mask shape = (B, 1, 1, S) or (B, 1, T, T) will get broad-casted (copied) as needed to match scores shape
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))

        # Step 3: Calculate the attention weights - how much should we attend to surrounding token representations
        attention_weights = self.softmax(scores)

        # Step 4: Not defined in the original paper apply dropout to attention weights as well
        attention_weights = self.attention_dropout(attention_weights)

        # Step 5: based on attention weights calculate new token representations
        # attention_weights shape = (B, NH, S, S)/(B, NH, T, T) or (B, NH, T, S), value shape = (B, NH, S/T, HD)
        # Final shape (B, NH, S, HD) for source MHAs or (B, NH, T, HD) target MHAs (again MHAs are used in 3 contexts)
        intermediate_token_representations = torch.matmul(attention_weights, value)

        return intermediate_token_representations, attention_weights  # attention weights for visualization purposes

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]

        # Step 1: Input linear projection
        # Notation: B - batch size, NH - number of heads, S/T - max src/trg token-sequence length, HD - head dimension
        # Shape goes from (B, S/T, NH*HD) over (B, S/T, NH, HD) to (B, NH, S/T, HD) (NH*HD=D where D is model dimension)
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]

        # Step 2: Apply attention - compare query with key and use that to combine values (see the function for details)
        intermediate_token_representations, attention_weights = self.attention(query, key, value, mask)

        # Potentially, for visualization purposes, log the attention weights, turn off during training though!
        # I had memory problems when I leave this on by default
        if self.log_attention_weights:
            self.attention_weights = attention_weights

        # Step 3: Reshape from (B, NH, S/T, HD) over (B, S/T, NH, HD) (via transpose) into (B, S/T, NHxHD) which is
        # the same shape as in the beginning of this forward function i.e. input to MHA (multi-head attention) module
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1, self.number_of_heads * self.head_dimension)

        # Step 4: Output linear projection
        token_representations = self.out_projection_net(reshaped)

        return token_representations


#
# Input modules
#


class Embedding(nn.Module):

    def __init__(self, vocab_size, model_dimension):
        super().__init__()
        self.embeddings_table = nn.Embedding(vocab_size, model_dimension)
        self.model_dimension = model_dimension

    def forward(self, token_ids_batch):
        assert token_ids_batch.ndim == 2, f'Expected: (batch size, max token sequence length), got {token_ids_batch.shape}'

        # token_ids_batch has shape (B, S/T), where B - batch size, S/T max src/trg token-sequence length
        # Final shape will be (B, S/T, D) where D is the model dimension, every token id has associated vector
        embeddings = self.embeddings_table(token_ids_batch)

        # (stated in the paper) multiply the embedding weights by the square root of model dimension
        # Page 5, Chapter 3.4 "Embeddings and Softmax"
        return embeddings * math.sqrt(self.model_dimension)


class PositionalEncoding(nn.Module):

    def __init__(self, model_dimension, dropout_probability, expected_max_sequence_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)

        # (stated in the paper) Use sine functions whose frequencies form a geometric progression as position encodings,
        # (learning encodings will also work so feel free to change it!). Page 6, Chapter 3.5 "Positional Encoding"
        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension)

        # Checkout playground.py for visualization of how these look like (it's super simple don't get scared)
        positional_encodings_table = torch.zeros(expected_max_sequence_length, model_dimension)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies)  # sine on even positions
        positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies)  # cosine on odd positions

        # Register buffer because we want to save the positional encodings table inside state_dict even though
        # these are not trainable (not model's parameters) so they otherwise would be excluded from the state_dict
        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

        # embedding_batch's shape = (B, S/T, D), where S/T max src/trg token-sequence length, D - model dimension
        # So here we get (S/T, D) shape which will get broad-casted to (B, S/T, D) when we try and add it to embeddings
        positional_encodings = self.positional_encodings_table[:embeddings_batch.shape[1]]

        # (stated in the paper) Applying dropout to the sum of positional encodings and token embeddings
        # Page 7, Chapter 5.4 "Regularization"
        return self.dropout(embeddings_batch + positional_encodings)


#
# Helper model functions
#


def get_clones(module, num_of_deep_copies):
    # Create deep copies so that we can tweak each module's weights independently
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])


# Count how many trainable weights the model has <- just for having a feeling for how big the model is
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_state_dict_shapes_and_names(model):
    # This part helped me figure out that I don't have positional encodings saved in the state dict
    print(model.state_dict().keys())

    # This part helped me see that src MHA was missing in the decoder since both it and trg MHA were referencing
    # the same MHA object in memory - stupid mistake, happens all the time, embrace the suck!
    for name, param in model.named_parameters():
        print(name, param.shape)
        if not param.requires_grad:
            raise Exception('Expected all of the params to be trainable - no param freezing used.')


# Testing the correctness of the transformer model - feel free to ignore - I used it during model development
if __name__ == "__main__":
    use_big_transformer = False

    # Dummy data
    src_vocab_size = 37000
    trg_vocab_size = 37000
    src_token_ids_batch = torch.randint(1, 10, size=(3, 2))
    trg_token_ids_batch = torch.randint(1, 10, size=(3, 2))

    transformer = Transformer(
        model_dimension=512,
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        number_of_heads=8,
        number_of_layers=6,
        dropout_probability=0.1
    )

    # These 2 functions helped me figure out the 2 bugs I had:
    # 1) I did not register positional encodings and thus they wouldn't be saved and later model-loading would fail
    # 2) I had a bug with MHA (attention) in decoder, where both src and trg were referencing the same MHA object in mem
    # It's a good practice to see whether the names, shapes and number of params make sense.
    # e.g. I knew that the big transformer had ~175 M params and I verified that here.
    analyze_state_dict_shapes_and_names(transformer)
    print(f'Size of the {"big" if use_big_transformer else "baseline"} transformer = {count_parameters(transformer)}')

    out = transformer(src_token_ids_batch, trg_token_ids_batch, src_mask=None, trg_mask=None)