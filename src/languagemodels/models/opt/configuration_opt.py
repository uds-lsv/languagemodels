# Implementation of ALiBi based on https://github.com/mosaicml/llm-foundry

from transformers.models.opt.configuration_opt import OPTConfig


class OPTWithALiBiConfig(OPTConfig):

    model_type = "OPTWithALiBi"

    def __init__(
            self, 
            vocab_size=50272, 
            hidden_size=768,
            num_hidden_layers=12, 
            ffn_dim=3072, 
            max_position_embeddings=2048,
            do_layer_norm_before=True, 
            _remove_final_layer_norm=False, 
            word_embed_proj_dim=None, 
            dropout=0.1, 
            attention_dropout=0, 
            num_attention_heads=12, 
            activation_function="relu", 
            layerdrop=0, 
            init_std=0.02, 
            use_cache=True, 
            pad_token_id=1, 
            bos_token_id=2, 
            eos_token_id=2, 
            enable_bias=True, 
            layer_norm_elementwise_affine=True,
            alibi=False,
            alibi_bias_max=8, 
            **kwargs
        ):

        super().__init__(
            vocab_size, 
            hidden_size, 
            num_hidden_layers, 
            ffn_dim, 
            max_position_embeddings, 
            do_layer_norm_before, 
            _remove_final_layer_norm, 
            word_embed_proj_dim, 
            dropout, 
            attention_dropout, 
            num_attention_heads, 
            activation_function, 
            layerdrop, 
            init_std, 
            use_cache, 
            pad_token_id, 
            bos_token_id, 
            eos_token_id, 
            enable_bias,
            layer_norm_elementwise_affine, 
            **kwargs
        )

        self.alibi = alibi
        self.alibi_bias_max = alibi_bias_max
