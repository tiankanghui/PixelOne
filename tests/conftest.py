import os

import pytest
import torch


@pytest.fixture(scope="session")
def processor():
    path = os.environ.get("PIXELONE_TEST_PROCESSOR")
    if not path:
        pytest.skip("Set PIXELONE_TEST_PROCESSOR to a released model directory for processor integration tests")
    from transformers import AutoProcessor
    from pixelone.processor import LegacyProcessorAdapter
    torch.set_num_threads(2)
    return LegacyProcessorAdapter(AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False))


def tiny_native(processor,moe=False,attention='sdpa',linear_head_dim=8):
    from videojev_hf.configuration_videojev import VideoJevConfig
    from videojev_hf.modeling_videojev import VideoJevForVideoUnderstanding
    torch.manual_seed(31)
    text=dict(vocab_size=len(processor.tokenizer),hidden_size=32,num_hidden_layers=2,num_attention_heads=2,
              num_key_value_heads=1,head_dim=16,layer_types=['linear_attention','full_attention'],
              linear_conv_kernel_dim=4,linear_key_head_dim=linear_head_dim,linear_value_head_dim=linear_head_dim,linear_num_key_heads=2,
              linear_num_value_heads=2,intermediate_size=64,
              rope_parameters=dict(rope_type='default',rope_theta=10000.,partial_rotary_factor=.5,mrope_section=[1,1,2]))
    if moe:text.update(moe_intermediate_size=16,shared_expert_intermediate_size=16,num_experts=4,num_experts_per_tok=2)
    cfg=VideoJevConfig(backbone_config=dict(model_type='qwen3_5_moe' if moe else 'qwen3_5',
        video_token_id=processor.video_token_id,image_token_id=processor.image_token_id,
        vision_start_token_id=processor.tokenizer.convert_tokens_to_ids('<|vision_start|>'),
        vision_end_token_id=processor.tokenizer.convert_tokens_to_ids('<|vision_end|>'),text_config=text,
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,out_hidden_size=32,
                           num_position_embeddings=16,patch_size=16,temporal_patch_size=2,spatial_merge_size=2)),
        head_dim=8,decide_token_id=processor.tokenizer.convert_tokens_to_ids('<|decide|>'),
        option_end_token_id=processor.tokenizer.convert_tokens_to_ids('<|option_end|>'))
    cfg._attn_implementation=attention
    return VideoJevForVideoUnderstanding(cfg)
