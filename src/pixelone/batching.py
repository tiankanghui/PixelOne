"""Pack variable-length decision states into one language-model forward."""

from contextlib import nullcontext

import torch


def collate_query_groups(groups, pad_token_id):
    """Pad input sequences and options while retaining each question's sample index."""
    if not groups:
        raise ValueError('Provide at least one query group')
    max_length = max(group['input_ids'].shape[1] for group in groups)
    max_options = max(group['pair_indices'].shape[1] for group in groups)
    input_ids = torch.full((len(groups), max_length), pad_token_id, dtype=torch.long)
    token_types = torch.zeros_like(input_ids)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    pairs, masks, queries, tasks, sample_indices = [], [], [], [], []
    for sample_index, group in enumerate(groups):
        if group['input_ids'].shape[0] != 1 or group['video_grid_thw'].shape[0] != 1:
            raise ValueError('Each input group must contain exactly one video')
        length = group['input_ids'].shape[1]
        input_ids[sample_index, :length] = group['input_ids'][0]
        token_types[sample_index, :length] = group['mm_token_type_ids'][0]
        attention_mask[sample_index, :length] = True
        padding = max_options - group['pair_indices'].shape[1]
        pairs.append(torch.nn.functional.pad(group['pair_indices'], (0, padding)))
        masks.append(torch.nn.functional.pad(group['score_mask'], (0, padding)))
        queries.append(group['query_indices'])
        tasks.append(group['decision_tasks'])
        sample_indices.extend([sample_index] * len(group['query_indices']))
    return dict(
        input_ids=input_ids,
        mm_token_type_ids=token_types,
        attention_mask=attention_mask,
        video_grid_thw=torch.cat([group['video_grid_thw'] for group in groups]),
        pair_indices=torch.cat(pairs),
        query_indices=torch.cat(queries),
        score_mask=torch.cat(masks),
        decision_tasks=torch.cat(tasks),
        sample_indices=torch.tensor(sample_indices),
    )


def forward_states(native, groups, features, labels=None):
    """Run packed decision questions and recover each state's scores and optional loss."""
    from pixelone.model import choice_loss

    if (
        not groups
        or len(groups) != len(features)
        or (labels is not None and len(labels) != len(groups))
    ):
        raise ValueError('Each training state needs its own group, features and labels')
    if any(value.is_inference() for value in features):
        raise ValueError('Clone frozen visual features outside inference_mode before backward')
    # Padding tokens never attend to or supply supervision for valid states.
    batch = collate_query_groups(groups, 0)
    device = native.get_input_embeddings().weight.device
    batch = {key: value.to(device) for key, value in batch.items()}
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    with amp:
        embeds = native.get_input_embeddings()(batch['input_ids'])
        if getattr(native, 'full_parameter_training', False) and device.type == 'cuda':
            embeds = embeds.to(torch.bfloat16)
        video_token = native.config.backbone_config.video_token_id
        all_slots = batch['input_ids'] == video_token
        for index, value in enumerate(features):
            slots = all_slots[index]
            if int(slots.sum()) != len(value):
                raise ValueError('Visual feature count differs from state tokens')
        # Checkpointing can mark embedding outputs as leaf tensors requiring grad.
        # Use the native out-of-place scatter, never mutate those leaves.
        embeds = embeds.masked_scatter(
            all_slots[..., None].expand_as(embeds), torch.cat(features).to(embeds)
        )
        if native.is_moe:
            positions, _ = native.backbone.get_rope_index(
                input_ids=batch['input_ids'],
                mm_token_type_ids=batch['mm_token_type_ids'],
                video_grid_thw=batch['video_grid_thw'],
                attention_mask=batch['attention_mask'],
            )
        else:
            positions = native.backbone.compute_3d_position_ids(
                input_ids=batch['input_ids'],
                inputs_embeds=embeds,
                video_grid_thw=batch['video_grid_thw'],
                attention_mask=batch['attention_mask'],
                mm_token_type_ids=batch['mm_token_type_ids'],
            )
        hidden = native.backbone.language_model(
            inputs_embeds=embeds,
            position_ids=positions,
            attention_mask=batch['attention_mask'],
            use_cache=False,
        ).last_hidden_state
    with torch.autocast(device_type=device.type, enabled=False):
        indices = batch['sample_indices']
        logits = native.head(
            hidden[indices[:, None], batch['pair_indices']], hidden[indices, batch['query_indices']]
        )
        mask = batch['score_mask']
        probs = (
            torch.where(
                (batch['decision_tasks'] == 1)[:, None],
                logits.masked_fill(~mask, -torch.inf).softmax(-1),
                logits.sigmoid(),
            )
            * mask
        )
        items = []
        losses = []
        offset = 0
        for index, group in enumerate(groups):
            count = len(group['query_indices'])
            options = group['pair_indices'].shape[1]
            item = dict(
                logits=logits[offset : offset + count, :options],
                probabilities=probs[offset : offset + count, :options],
                score_mask=mask[offset : offset + count, :options],
            )
            if labels is not None:
                losses.append(
                    choice_loss(item['logits'], item['score_mask'], labels[index].to(device))
                )
            items.append(item)
            offset += count
        result = dict(items=items)
        if losses:
            result['loss'] = torch.stack(losses).mean()
    return result
