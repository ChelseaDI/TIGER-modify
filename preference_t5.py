"""Wrap a pretrained T5 with a projection and encoder input concatenation."""
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSeq2SeqLM


class PreferenceT5(nn.Module):
    def __init__(self, t5, content_dim):
        super().__init__()
        # The supplied T5 keeps all its pretrained parameters.
        self.t5 = t5
        self.config = t5.config
        self.config.content_dim = content_dim
        self.config.use_group_preferences = True
        self.preference_projection = nn.Linear(content_dim, self.config.d_model, bias=False)
        nn.init.normal_(self.preference_projection.weight, std=content_dim ** -0.5)

    def forward(self, input_ids, attention_mask, group_preferences,
                preference_mask, labels=None):
        """Training: combine inputs and let T5 compute the generation loss."""
        # Convert SID IDs and group preferences to the same embedding dimension.
        sid_embeddings = self.t5.get_input_embeddings()(input_ids)
        preferences = self.preference_projection(group_preferences.to(sid_embeddings.dtype))

        # Append preference vectors and their padding mask to the SID sequence.
        embeddings = torch.cat([sid_embeddings, preferences], dim=1)
        mask = torch.cat([attention_mask, preference_mask], dim=1)

        return self.t5(
            inputs_embeds=embeddings,
            attention_mask=mask,
            labels=labels,
        )

    def encode_preferences(self, input_ids, attention_mask, group_preferences,
                           preference_mask):
        """Evaluation: encode the combined input once before beam search."""
        sid_embeddings = self.t5.get_input_embeddings()(input_ids)
        preferences = self.preference_projection(group_preferences.to(sid_embeddings.dtype))
        embeddings = torch.cat([sid_embeddings, preferences], dim=1)
        mask = torch.cat([attention_mask, preference_mask], dim=1)

        encoder_outputs = self.t5.get_encoder()(
            inputs_embeds=embeddings,
            attention_mask=mask,
            return_dict=True,
        )
        return encoder_outputs, mask

    def generate(self, **kwargs):
        return self.t5.generate(**kwargs)

    def save_pretrained(self, directory):
        # Store the complete wrapper, including T5 and the projection.
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(directory)
        torch.save(self.state_dict(), str(Path(directory) / 'pytorch_model.bin'))

    @classmethod
    def from_pretrained(cls, directory):
        config = AutoConfig.from_pretrained(directory)
        t5 = AutoModelForSeq2SeqLM.from_config(config)
        model = cls(t5, config.content_dim)
        state = torch.load(str(Path(directory) / 'pytorch_model.bin'), map_location='cpu')
        model.load_state_dict(state)
        return model
