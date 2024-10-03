import torch
import torch.nn as nn
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SubModules.encoder import Encoder
from SubModules.decoder import Decoder
from configs.config import CONFIG
class Transformer(nn.Module):
    def __init__(self, hidden_dims, num_encoder_layers, num_decoder_layers, vocab_size, sos_idx, eos_idx, pad_idx, device, max_seq_length_encoder=512, max_seq_length_decoder=128):
        super(Transformer, self).__init__()
        self.device = device
        self.hidden_dims = hidden_dims
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        
        # Encoder and Decoder
        self.encoder = Encoder(hidden_dims, num_encoder_layers)
        self.decoder = Decoder(hidden_dims, num_decoder_layers)
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dims)
        
        # Final linear layer to map decoder output to vocab size
        self.fc_out = nn.Linear(hidden_dims, vocab_size)
        self.init_weights()
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.max_seq_length_decoder = max_seq_length_decoder
        self.max_seq_length_encoder = max_seq_length_encoder

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):
        # Embed the input_ sequence
        src_emb = self.embedding(src)
        
        # Create padding mask
        src_pad_mask = (src == self.pad_idx).unsqueeze(1).unsqueeze(2)

        # Encoder forward pass
        encoder_output = self.encoder(src_emb,mask=src_pad_mask)

        if tgt is not None:
            # Training mode
            max_len = min(tgt.size(1), CONFIG["max_seq_length_decoder"])
            tgt_vocab_size = self.fc_out.out_features
            outputs = torch.zeros(src.size(0), max_len, tgt_vocab_size).to(self.device)

            # First input to the decoder is the SOS token
            input_ = torch.full((src.size(0), 1), self.sos_idx, dtype=torch.long, device=self.device)

            for t in range(1, max_len):
                tgt_emb = self.embedding(input_)
                decoder_output = self.decoder(encoder_output, tgt_emb)
                prediction = self.fc_out(decoder_output[:, -1])
                outputs[:, t] = prediction

                # Teacher forcing
                teacher_force = random.random() < teacher_forcing_ratio
                top1 = prediction.argmax(1)
                input_ = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        else:
            # Inference mode
            max_len = self.max_seq_length_decoder
            outputs = torch.zeros(src.size(0), max_len, self.fc_out.out_features).to(self.device)
            input_ = torch.full((src.size(0), 1), self.sos_idx, dtype=torch.long, device=self.device)

            for t in range(1, max_len):
                tgt_emb = self.embedding(input_)
                decoder_output = self.decoder(encoder_output, tgt_emb)
                prediction = self.fc_out(decoder_output[:, -1])
                outputs[:, t] = prediction
                input_ = prediction.argmax(1).unsqueeze(1)

                if (input_ == self.eos_idx).all():
                    break
        
        return outputs
