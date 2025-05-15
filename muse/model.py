from copy import deepcopy

import mido
import torch
from torch import nn

from muse.data import AudioProcessor, DataProcessor, Tokeniser
from muse.utils import count_trainable_params


def get_decoder_inputs_and_targets(
    y: torch.Tensor,
    eos_id: int,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Want the model to predict all but the last token and so we slice until it
    # Replace <eos> with <pad> also as it has to predict it
    text_inputs = deepcopy(y)[:, :-1]
    text_inputs[text_inputs == eos_id] = pad_id

    # Targets are all but the first token (don't predict <bos>)
    text_targets = deepcopy(y)[:, 1:]
    return text_inputs, text_targets


def calculate_accuracy(scores, y, pad_token_id=None):
    # scores: B, N, N_classes
    # y: B, N
    if pad_token_id is not None:
        # Remove padding tokens from the target
        mask = y != pad_token_id
        scores = scores[mask]
        y = y[mask]
    correct = torch.sum(scores.argmax(dim=-1) == y)
    return correct/y.numel(), correct


def make_causal_mask(seq_len: int, device=torch.device):
    """ Text tokens (of sequence_len) cannot attend to future tokens
    """
    # Shape: (seq_len, seq_len)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask  # Additive mask for attention scores


class MusicTranscriber(nn.Module):
    def __init__(self, model_dict):
        """ Standard encoder-decoder architecture
        """
        super().__init__()
        self.d_model = model_dict.get('d_model')
        self.enc_dims = model_dict.get('enc_dims')
        self.enc_max_len = model_dict.get('enc_max_len')
        self.dec_max_len = model_dict.get('dec_max_len')
        self.dec_vocab_size = model_dict.get('dec_vocab_size')

        self.eos_id = model_dict.get('eos_id')
        self.bos_id = model_dict.get('bos_id')
        self.pad_id = model_dict.get('pad_id')

        self.transformer_model = nn.Transformer(
            d_model=self.d_model, batch_first=True)

        # Initialise positional encodings
        self.encoder_pos_encodings = nn.Parameter(
            torch.empty(1, self.enc_max_len, self.d_model))
        nn.init.normal_(self.encoder_pos_encodings, std=0.02)
        self.decoder_pos_embeddings = nn.Parameter(
            torch.empty(1, self.dec_max_len, self.d_model))
        nn.init.normal_(self.decoder_pos_embeddings, std=0.02)

        self.encoder_mlp = nn.Linear(self.enc_dims, self.d_model)
        self.decoder_emb = nn.Embedding(
            self.dec_vocab_size, self.d_model, padding_idx=self.pad_id)
        self.decoder_classifier = nn.Linear(self.d_model, self.dec_vocab_size)

    def forward(self, x, y):
        B, seq_len = y.shape
        _, enc_seq_len, _ = x.shape

        tgt_key_padding_mask = (y == self.pad_id)
        # tgt_key_padding_mask = None
        # Positional embeddings
        x_embs = self.encoder_mlp(
            x) + self.encoder_pos_encodings[:, :enc_seq_len, :]
        y_embs = self.decoder_emb(y) + self.decoder_pos_embeddings[:, :seq_len, :]

        # Prevents attention with padding tokens

        # Causal mask to prevent attending to future tokens
        causal_mask = make_causal_mask(seq_len, device=y_embs.device)
        res = self.transformer_model(
            x_embs,
            y_embs,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        scores = self.decoder_classifier(res)
        return scores

    def trainable_params(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    # def forward_sequential(self, x: torch.Tensor):
    #     # assert type(x) is torch.Tensor
    #     generated = torch.tensor(
    #         [[self.bos_id]], dtype=torch.int32, device=x.device,
    #     )
    #     for _ in range(self.text_seq_len_max - generated.size(1)):
    #         # assume outputs.logits [1, T, V]
    #         test_scores = self.forward(x, generated)
    #         # 3) Greedy pick at last position
    #         next_token = torch.argmax(
    #             test_scores[:, -1, :], dim=-1, keepdim=True,
    #         )  # [1,1]
    #         # 4) Append and check EOS
    #         generated = torch.cat([generated, next_token], dim=1)  # [1, T+1]
    #         if next_token.item() == self.eos_id:
    #             break
    #     return generated


if __name__ == '__main__':
    # Prepare data

    midi_path = '/Users/kenton/Desktop/2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.midi'
    wav_path = '/Users/kenton/Desktop/2008/MIDI-Unprocessed_01_R1_2008_01-04_ORIG_MID--AUDIO_01_R1_2008_wav--1.wav'

    # # Load the MIDI file

    dp = DataProcessor(16)

    spectrogram, sr = dp.make_spectrogram(wav_path)
    midi_file = mido.MidiFile(midi_path)
    midi_processed = dp.process_midi(midi_file)
    spectrograms, inputs = dp.collate_fn(spectrogram, midi_processed)

    model_dict = {
        'd_model': 512,
        'enc_dims': dp.ap.n_mels,
        'enc_max_len': dp.max_enc_len,
        'dec_max_len': dp.max_dec_len,
        'dec_vocab_size': dp.tok.vocab_size,
        'bos_id': dp.tok.bos_id,
        'eos_id': dp.tok.eos_id,
        'pad_id': dp.tok.pad_id,
    }

    mt = MusicTranscriber(model_dict)
    res = mt(spectrograms, inputs)
    print(spectrograms.shape)
    print(inputs.shape)
    print(res.shape)

    print(count_trainable_params(mt))
