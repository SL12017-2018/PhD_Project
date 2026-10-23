#!/usr/bin/env python3
"""
Custom collate function for IEMOCAP data
"""

import torch
from typing import List, Dict, Any

def iemocap_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for IEMOCAP dataset"""
    
    # Handle tensor fields that need proper batching
    collated = {}
    
    # Simple fields that can be stacked or converted to tensors
    tensor_fields = ['audio', 'label', 'dialog_id', 'turn_id', 'labels', 'speaker_ids']
    
    for field in tensor_fields:
        if field in batch[0]:
            values = [item[field] for item in batch]
            if field in ['label', 'labels', 'dialog_id', 'turn_id', 'speaker_ids']:
                # Convert scalars to tensor
                collated[field] = torch.tensor(values, dtype=torch.long)
            else:
                # Stack tensors
                collated[field] = torch.stack([torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in values])
    
    # Handle nested dictionary fields (text_inputs, audio_inputs)
    dict_fields = ['text_inputs', 'audio_inputs']
    
    for field in dict_fields:
        if field in batch[0] and isinstance(batch[0][field], dict):
            collated[field] = {}
            for key in batch[0][field].keys():
                values = [item[field][key] for item in batch]
                if isinstance(values[0], torch.Tensor):
                    collated[field][key] = torch.stack(values)
                else:
                    collated[field][key] = torch.tensor(values)
    
    # Handle string/list fields that don't need tensor conversion
    string_fields = ['text', 'speaker', 'emotion', 'file', 'gender', 'context']
    
    for field in string_fields:
        if field in batch[0]:
            collated[field] = [item[field] for item in batch]
    
    return collated
