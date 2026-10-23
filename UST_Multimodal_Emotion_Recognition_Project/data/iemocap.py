"""
IEMOCAP Dataset Loader for MemoCMT
Handles text + audio multimodal data with dialogue context
"""

import os
import json
import torch
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from datasets import load_from_disk, Dataset as HFDataset
import warnings
warnings.filterwarnings("ignore")


class IEMOCAPDataset(Dataset):
    """IEMOCAP Dataset for multimodal emotion recognition"""
    
    def __init__(
        self,
        root_path: str,
        split: str = "train",
        sample_rate: int = 16000,
        max_audio_len_s: float = 7.0,
        max_length: int = 256,  # Text max length for compatibility
        max_audio_length: Optional[float] = None,  # Alternative name for max_audio_len_s
        context_window: int = 5,
        emotion_mapping: Optional[Dict] = None
    ):
        self.root_path = Path(root_path)
        self.split = split
        self.sample_rate = sample_rate
        self.max_length = max_length  # For text tokenization compatibility
        
        # Handle audio length parameters (max_audio_length takes precedence if provided)
        if max_audio_length is not None:
            self.max_audio_len_s = max_audio_length
        else:
            self.max_audio_len_s = max_audio_len_s
        
        self.max_audio_len = int(sample_rate * self.max_audio_len_s)
        self.context_window = context_window
        
        # IEMOCAP emotion mapping (6-class mapping for compatibility)
        if emotion_mapping is None:
            self.emotion_mapping = {
                'angry': 0,
                'happy': 1,
                'sad': 2,
                'neutral': 3,
                'excited': 4,
                'frustrated': 5,
                # Map additional emotions to existing categories
                'surprise': 1,  # Map surprise to happy (both positive)
                'fear': 2,      # Map fear to sad (both negative)
                'disgust': 0,   # Map disgust to angry (both negative arousal)
                'other': 3      # Map other to neutral
            }
        else:
            self.emotion_mapping = emotion_mapping
            
        self.num_classes = len(set(self.emotion_mapping.values()))
        
        # Initialize tokenizer and audio processor for model compatibility
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            self.tokenizer = None
            
        try:
            self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        except Exception as e:
            print(f"Warning: Could not load audio processor: {e}")
            self.audio_processor = None
        
        # Create speaker mapping for consistent IDs
        self.speaker_to_id = {'M': 0, 'F': 1}  # Male=0, Female=1 for IEMOCAP
        
        # Load data using CSV export format if available, otherwise HuggingFace
        self.data = self._load_csv_export() or self._load_hf_dataset()
        print(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_hf_dataset(self) -> List[Dict]:
        """Load data from HuggingFace dataset format"""
        try:
            # Try to load from HuggingFace dataset format
            dataset_path = self.root_path
            if (dataset_path / "dataset_dict.json").exists():
                print(f"Loading HuggingFace dataset from {dataset_path}")
                dataset = load_from_disk(str(dataset_path))
                
                # Get the split data
                if self.split in dataset:
                    hf_data = dataset[self.split]
                elif 'train' in dataset:
                    # Use train data and split it
                    hf_data = dataset['train']
                    hf_data = self._split_dataset(hf_data)
                else:
                    raise ValueError(f"Split '{self.split}' not found in dataset")
                
                # Convert to our format
                data_list = []
                for i, item in enumerate(hf_data):
                    try:
                        # Extract file info for dialog and turn IDs
                        file_name = item.get('file', f'unknown_{i}')
                        
                        # Parse dialog info from filename (IEMOCAP format)
                        dialog_id, turn_id, speaker = self._parse_filename(file_name)
                        
                        # Get emotion label
                        major_emotion = item.get('major_emotion', 'neutral').lower()
                        if major_emotion in self.emotion_mapping:
                            label = self.emotion_mapping[major_emotion]
                        else:
                            # Skip unknown emotions or map to neutral
                            label = self.emotion_mapping.get('neutral', 0)
                        
                        # Get text
                        text = item.get('transcription', '').strip()
                        if not text:
                            continue  # Skip empty transcriptions
                        
                        # Get audio - try multiple methods
                        audio_array = None
                        audio_sr = 16000
                        
                        # Method 1: Try exported audio files (numbered)
                        audio_export_path = "/home/BTECH_7TH_SEM/Desktop/MML/iemocap_export/train/audio"
                        if os.path.exists(audio_export_path):
                            audio_file = os.path.join(audio_export_path, f"{i}.wav")
                            if os.path.exists(audio_file):
                                try:
                                    import soundfile as sf
                                    audio_array, audio_sr = sf.read(audio_file)
                                    if len(audio_array) == 0:
                                        audio_array = None
                                except Exception as e:
                                    print(f"Warning: Could not load exported audio {audio_file}: {e}")
                        
                        # Method 2: Try HuggingFace dataset audio field
                        if audio_array is None:
                            audio_data = item.get('audio', {})
                            if isinstance(audio_data, dict) and 'array' in audio_data:
                                try:
                                    audio_array = np.array(audio_data['array'], dtype=np.float32)
                                    audio_sr = audio_data.get('sampling_rate', 16000)
                                    if len(audio_array) == 0:
                                        audio_array = None
                                except:
                                    audio_array = None
                        
                        # Skip if no audio found
                        if audio_array is None or len(audio_array) == 0:
                            if i < 10:  # Only print warnings for first few items
                                print(f"Warning: No audio data for item {i}")
                            continue
                        
                        # Resample if necessary
                        if audio_sr != self.sample_rate:
                            audio_array = librosa.resample(
                                audio_array, 
                                orig_sr=audio_sr, 
                                target_sr=self.sample_rate
                            )
                        
                        data_entry = {
                            "dialog_id": dialog_id,
                            "turn_id": turn_id,
                            "speaker": speaker,
                            "text": text,
                            "audio": audio_array,
                            "emotion": major_emotion,
                            "label": label,
                            "file": file_name,
                            # Additional features
                            "gender": item.get('gender', 'unknown'),
                            "speaking_rate": item.get('speaking_rate', 0.0),
                            "pitch_mean": item.get('pitch_mean', 0.0),
                            "rms": item.get('rms', 0.0)
                        }
                        
                        data_list.append(data_entry)
                        
                    except Exception as e:
                        print(f"Error processing item {i}: {e}")
                        continue
                
                return data_list
                
        except Exception as e:
            print(f"Error loading HuggingFace dataset: {e}")
            return self._load_manifest()
    
    def _load_csv_export(self) -> List[Dict]:
        """Load IEMOCAP data from CSV export format"""
        try:
            # Check for CSV export format
            csv_path = "/home/BTECH_7TH_SEM/Desktop/MML/iemocap_export/train/metadata/train_metadata.csv"
            audio_dir = "/home/BTECH_7TH_SEM/Desktop/MML/iemocap_export/train/audio"
            
            if not os.path.exists(csv_path) or not os.path.exists(audio_dir):
                print(f"CSV export not found at {csv_path} or {audio_dir}")
                return []
            
            print(f"Loading CSV export from {csv_path}")
            
            # Read CSV metadata
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            data_list = []
            for idx, row in df.iterrows():
                try:
                    # Map CSV row index to audio file (0.wav, 1.wav, etc.)
                    audio_file = os.path.join(audio_dir, f"{idx}.wav")
                    
                    if not os.path.exists(audio_file):
                        print(f"Warning: Audio file not found: {audio_file}")
                        continue
                    
                    # Load audio
                    import soundfile as sf
                    audio_array, audio_sr = sf.read(audio_file)
                    if len(audio_array) == 0:
                        continue
                    
                    # Resample if necessary
                    if audio_sr != self.sample_rate:
                        audio_array = librosa.resample(
                            audio_array, 
                            orig_sr=audio_sr, 
                            target_sr=self.sample_rate
                        )
                    
                    # Extract file info for dialog and turn IDs
                    file_name = row['file'].strip()
                    dialog_id, turn_id, speaker = self._parse_filename(file_name)
                    
                    # Get emotion label with validation
                    major_emotion = row['major_emotion'].strip().lower()
                    if major_emotion in self.emotion_mapping:
                        label = self.emotion_mapping[major_emotion]
                    else:
                        print(f"Warning: Unknown emotion '{major_emotion}' mapped to neutral")
                        label = self.emotion_mapping.get('neutral', 3)
                    
                    # Ensure label is within valid range [0, 5]
                    if not (0 <= label <= 5):
                        print(f"Warning: Invalid label {label} for emotion '{major_emotion}', using neutral")
                        label = 3
                    
                    # Get text
                    text = row['transcription'].strip()
                    if not text:
                        continue
                    
                    data_entry = {
                        "dialog_id": dialog_id,
                        "turn_id": turn_id,
                        "speaker": speaker,
                        "text": text,
                        "audio": audio_array.astype(np.float32),
                        "emotion": major_emotion,
                        "label": label,
                        "file": file_name,
                        # Additional features from CSV
                        "gender": row['gender'].strip() if pd.notna(row['gender']) else 'unknown',
                        "speaking_rate": float(row['speaking_rate']) if pd.notna(row['speaking_rate']) else 0.0,
                        "pitch_mean": float(row['pitch_mean']) if pd.notna(row['pitch_mean']) else 0.0,
                        "rms": float(row['rms']) if pd.notna(row['rms']) else 0.0,
                        # Emotion probabilities
                        "frustrated": float(row['frustrated']) if pd.notna(row['frustrated']) else 0.0,
                        "angry": float(row['angry']) if pd.notna(row['angry']) else 0.0,
                        "sad": float(row['sad']) if pd.notna(row['sad']) else 0.0,
                        "happy": float(row['happy']) if pd.notna(row['happy']) else 0.0,
                        "excited": float(row['excited']) if pd.notna(row['excited']) else 0.0,
                        "fear": float(row['fear']) if pd.notna(row['fear']) else 0.0,
                        "surprise": float(row['surprise']) if pd.notna(row['surprise']) else 0.0,
                        "disgust": float(row['disgust']) if pd.notna(row['disgust']) else 0.0,
                        "neutral": float(row['neutral']) if pd.notna(row['neutral']) else 0.0,
                    }
                    
                    data_list.append(data_entry)
                    
                    # Process all samples (removed limit for full dataset)
                    
                except Exception as e:
                    print(f"Error processing CSV row {idx}: {e}")
                    continue
            
            print(f"Successfully loaded {len(data_list)} samples from CSV export")
            return data_list
            
        except Exception as e:
            print(f"Error loading CSV export: {e}")
            return []
    
    def _parse_filename(self, filename: str) -> Tuple[int, int, str]:
        """Parse IEMOCAP filename to extract dialog_id, turn_id, speaker"""
        try:
            # IEMOCAP format: Ses01F_impro01_F000.wav or similar
            if '_' in filename:
                parts = filename.replace('.wav', '').split('_')
                if len(parts) >= 3:
                    session_part = parts[0]  # Ses01F
                    dialog_part = parts[1]   # impro01
                    turn_part = parts[2]     # F000
                    
                    # Extract session number
                    session_num = ''.join(filter(str.isdigit, session_part))
                    session_num = int(session_num) if session_num else 1
                    
                    # Create dialog ID from session and dialog type
                    dialog_type = dialog_part[:6] if len(dialog_part) >= 6 else dialog_part
                    dialog_id = hash(f"{session_num}_{dialog_type}") % 10000
                    
                    # Extract turn ID and speaker
                    speaker = turn_part[0] if turn_part else 'A'
                    turn_num = ''.join(filter(str.isdigit, turn_part))
                    turn_id = int(turn_num) if turn_num else 0
                    
                    return dialog_id, turn_id, speaker
            
            # Fallback for non-standard filenames
            file_hash = hash(filename) % 10000
            return file_hash, 0, 'A'
            
        except Exception as e:
            print(f"Error parsing filename {filename}: {e}")
            return hash(filename) % 10000, 0, 'A'
    
    def _split_dataset(self, hf_data) -> HFDataset:
        """Split dataset into train/val/test if only train split exists"""
        total_size = len(hf_data)
        
        if self.split == "train":
            # Use first 80% for training
            end_idx = int(0.8 * total_size)
            return hf_data.select(range(end_idx))
        elif self.split == "val":
            # Use next 10% for validation
            start_idx = int(0.8 * total_size)
            end_idx = int(0.9 * total_size)
            return hf_data.select(range(start_idx, end_idx))
        elif self.split == "test":
            # Use last 10% for testing
            start_idx = int(0.9 * total_size)
            return hf_data.select(range(start_idx, total_size))
        else:
            return hf_data
    
    def _load_manifest(self) -> List[Dict]:
        """Load data manifest from JSON files"""
        manifest_file = self.root_path / f"{self.split}_meta.json"
        
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                return json.load(f)
        else:
            # Create manifest from existing data structure
            return self._create_manifest()
    
    def _create_manifest(self) -> List[Dict]:
        """Create manifest from existing IEMOCAP data structure"""
        manifest = []
        
        # Look for existing data files
        data_paths = [
            self.root_path / "train",
            self.root_path.parent / "iemocap_export" / "train"
        ]
        
        for data_path in data_paths:
            if data_path.exists():
                audio_dir = data_path / "audio"
                metadata_dir = data_path / "metadata"
                
                if audio_dir.exists() and metadata_dir.exists():
                    manifest = self._parse_iemocap_structure(audio_dir, metadata_dir)
                    break
        
        # If no structured data, create from CSV if available
        if not manifest:
            manifest = self._create_from_csv()
        
        # Save manifest for future use
        manifest_file = self.root_path / f"{self.split}_meta.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest
    
    def _parse_iemocap_structure(self, audio_dir: Path, metadata_dir: Path) -> List[Dict]:
        """Parse structured IEMOCAP data"""
        manifest = []
        
        # Find all audio files
        audio_files = list(audio_dir.glob("*.wav"))
        
        for audio_file in audio_files:
            # Extract utterance info from filename
            # IEMOCAP format: Ses01F_impro01_F000.wav
            filename = audio_file.stem
            parts = filename.split('_')
            
            if len(parts) >= 3:
                session = parts[0][:5]  # Ses01
                dialog_type = parts[1]  # impro01 or script01
                utt_id = parts[2]  # F000
                
                # Extract speaker and emotion from utt_id
                speaker = utt_id[0]  # F or M
                
                # Look for corresponding metadata
                metadata_file = metadata_dir / f"{filename}.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    text = metadata.get('text', '')
                    emotion = metadata.get('emotion', 'neu')
                    
                    # Map emotion to label
                    if emotion in self.emotion_mapping:
                        label = self.emotion_mapping[emotion]
                        
                        manifest.append({
                            "dialog_id": hash(f"{session}_{dialog_type}") % 10000,
                            "turn_id": int(utt_id[1:]) if utt_id[1:].isdigit() else 0,
                            "speaker": speaker,
                            "text": text,
                            "audio_path": str(audio_file),
                            "emotion": emotion,
                            "label": label
                        })
        
        return manifest
    
    def _create_from_csv(self) -> List[Dict]:
        """Create manifest from CSV data if available"""
        manifest = []
        
        # Look for CSV files in common locations
        csv_paths = [
            self.root_path / f"{self.split}.csv",
            self.root_path.parent / f"iemocap_{self.split}.csv"
        ]
        
        for csv_path in csv_paths:
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                
                for idx, row in df.iterrows():
                    if 'text' in row and 'emotion' in row:
                        emotion = row['emotion'].lower()[:3]  # First 3 chars
                        
                        if emotion in self.emotion_mapping:
                            manifest.append({
                                "dialog_id": row.get('dialog_id', idx // 10),
                                "turn_id": row.get('turn_id', idx),
                                "speaker": row.get('speaker', 'A'),
                                "text": str(row['text']),
                                "audio_path": row.get('audio_path', ''),
                                "emotion": emotion,
                                "label": self.emotion_mapping[emotion]
                            })
                break
        
        return manifest
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample with dialogue context"""
        item = self.data[idx]
        
        # Get current utterance
        current_text = item['text']
        current_audio = item['audio']  # Already loaded as numpy array
        current_label = item['label']
        current_speaker = item['speaker']
        dialog_id = item['dialog_id']
        turn_id = item['turn_id']
        
        # Process audio
        audio_waveform = self._process_audio(current_audio)
        
        # Get dialogue context
        context_items = self._get_dialogue_context(dialog_id, turn_id)
        
        # Tokenize text using the text encoder's tokenizer
        if hasattr(self, 'tokenizer'):
            text_inputs = self.tokenizer(
                current_text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            # Remove batch dimension
            text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        else:
            # Fallback - just provide the raw text
            text_inputs = {"raw_text": current_text}
        
        # Process audio for the audio encoder
        if hasattr(self, 'audio_processor'):
            audio_inputs = self.audio_processor(
                audio_waveform.numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding='max_length',
                max_length=self.max_audio_len
            )
            # Remove batch dimension
            audio_inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v 
                           for k, v in audio_inputs.items()}
        else:
            # Fallback - just provide the processed waveform
            audio_inputs = {"input_values": audio_waveform}
        
        return {
            # Original format for compatibility
            "text": current_text,
            "audio": audio_waveform,
            "label": current_label,
            "speaker": current_speaker,
            "dialog_id": dialog_id,
            "turn_id": turn_id,
            "context": context_items,
            "emotion": item['emotion'],
            "file": item['file'],
            "gender": item.get('gender', 'unknown'),
            
            # Model-expected format
            "text_inputs": text_inputs,
            "audio_inputs": audio_inputs,
            "labels": current_label,  # Model expects 'labels' not 'label'
            "speaker_ids": torch.tensor(self.speaker_to_id.get(current_speaker, 0), dtype=torch.long)  # Map speaker to correct ID
        }
    
    def _process_audio(self, audio_array: np.ndarray) -> torch.Tensor:
        """Process audio array to fixed length tensor"""
        try:
            # Ensure audio is float32
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Pad or truncate to fixed length
            if len(audio_array) > self.max_audio_len:
                audio_array = audio_array[:self.max_audio_len]
            else:
                audio_array = np.pad(audio_array, (0, self.max_audio_len - len(audio_array)))
            
            return torch.FloatTensor(audio_array)
        
        except Exception as e:
            print(f"Error processing audio: {e}")
            return torch.zeros(self.max_audio_len)
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio"""
        if not audio_path or not os.path.exists(audio_path):
            # Return silence if no audio
            return torch.zeros(self.max_audio_len)
        
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Pad or truncate to fixed length
            if len(audio) > self.max_audio_len:
                audio = audio[:self.max_audio_len]
            else:
                audio = np.pad(audio, (0, self.max_audio_len - len(audio)))
            
            return torch.FloatTensor(audio)
        
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return torch.zeros(self.max_audio_len)
    
    def _get_dialogue_context(self, dialog_id: int, turn_id: int) -> List[Dict]:
        """Get surrounding utterances in dialogue for context"""
        context_items = []
        
        # Find all utterances in the same dialogue
        dialogue_utts = [
            item for item in self.data 
            if item['dialog_id'] == dialog_id
        ]
        
        # Sort by turn_id
        dialogue_utts.sort(key=lambda x: x['turn_id'])
        
        # Find current position
        current_pos = None
        for i, utt in enumerate(dialogue_utts):
            if utt['turn_id'] == turn_id:
                current_pos = i
                break
        
        if current_pos is not None:
            # Get context window
            start_pos = max(0, current_pos - self.context_window)
            end_pos = min(len(dialogue_utts), current_pos + self.context_window + 1)
            
            context_items = dialogue_utts[start_pos:end_pos]
        
        return context_items


def collate_dialogue_batch(batch: List[Dict], tokenizer, processor) -> Dict:
    """Collate function for dialogue batches"""
    
    # Separate current utterances and contexts
    texts = [item['text'] for item in batch]
    audios = torch.stack([item['audio'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    speakers = [item['speaker'] for item in batch]
    dialog_ids = torch.tensor([item['dialog_id'] for item in batch])
    turn_ids = torch.tensor([item['turn_id'] for item in batch])
    
    # Tokenize texts
    text_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Process audio
    audio_inputs = processor(
        audios.numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    
    # Process context (simplified - take last N utterances)
    context_texts = []
    context_speakers = []
    
    for item in batch:
        context = item['context']
        if context:
            # Take last few context utterances
            ctx_texts = [ctx['text'] for ctx in context[-3:]]  # Last 3 context utterances
            ctx_speakers = [ctx['speaker'] for ctx in context[-3:]]
        else:
            ctx_texts = [""]
            ctx_speakers = [""]
        
        context_texts.append(" [SEP] ".join(ctx_texts))
        context_speakers.append(ctx_speakers)
    
    # Tokenize context
    context_inputs = tokenizer(
        context_texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    
    # Speaker mapping
    speaker_mapping = {'F': 0, 'M': 1, 'A': 0, 'B': 1}  # Flexible mapping
    speaker_ids = torch.tensor([speaker_mapping.get(s, 0) for s in speakers])
    
    return {
        "text_inputs": text_inputs,
        "audio_inputs": audio_inputs,
        "context_inputs": context_inputs,
        "labels": labels,
        "speaker_ids": speaker_ids,
        "dialog_ids": dialog_ids,
        "turn_ids": turn_ids,
        "raw_texts": texts,
        "raw_speakers": speakers
    }


def create_dataloaders(
    root_path: str,
    tokenizer,
    processor,
    batch_size: int = 8,
    num_workers: int = 4,
    sample_rate: int = 16000
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders"""
    
    # Create datasets
    train_dataset = IEMOCAPDataset(
        root_path=root_path,
        split="train",
        sample_rate=sample_rate
    )
    
    val_dataset = IEMOCAPDataset(
        root_path=root_path,
        split="val",
        sample_rate=sample_rate
    )
    
    test_dataset = IEMOCAPDataset(
        root_path=root_path,
        split="test",
        sample_rate=sample_rate
    )
    
    # Create collate function with tokenizer and processor
    def collate_fn(batch):
        return collate_dialogue_batch(batch, tokenizer, processor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print(f"  Batch size: {batch_size}")
    print(f"  Num classes: {train_dataset.num_classes}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
    
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Test with existing IEMOCAP data
    root_path = "../iemocap_dataset"
    
    train_loader, val_loader, test_loader = create_dataloaders(
        root_path=root_path,
        tokenizer=tokenizer,
        processor=processor,
        batch_size=4
    )
    
    # Test loading a batch
    for batch in train_loader:
        print("Batch keys:", batch.keys())
        print("Text inputs shape:", batch['text_inputs']['input_ids'].shape)
        print("Audio inputs shape:", batch['audio_inputs']['input_values'].shape)
        print("Labels shape:", batch['labels'].shape)
        print("Sample text:", batch['raw_texts'][0])
        break


def create_dummy_batch(
    tokenizer,
    processor, 
    batch_size: int = 2,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Create a dummy batch for testing purposes
    
    Args:
        tokenizer: Text tokenizer
        processor: Audio processor
        batch_size: Size of dummy batch
        device: Device to place tensors on
    
    Returns:
        Dictionary containing dummy batch data
    """
    
    import numpy as np
    
    # Dummy text data
    dummy_texts = [
        "I am feeling very happy today!",
        "This situation makes me quite sad.",
        "I'm angry about what happened.",
        "Everything seems neutral to me."
    ]
    
    # Select texts for batch
    texts = (dummy_texts * ((batch_size // len(dummy_texts)) + 1))[:batch_size]
    
    # Tokenize texts
    text_encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    
    # Create dummy audio (1 second of random noise)
    sample_rate = 16000
    audio_length = sample_rate  # 1 second
    
    dummy_audios = []
    for _ in range(batch_size):
        # Generate random audio-like signal
        audio = np.random.randn(audio_length).astype(np.float32) * 0.1
        dummy_audios.append(audio)
    
    # Process audio
    audio_inputs = processor(
        dummy_audios,
        sampling_rate=sample_rate,
        return_tensors='pt',
        padding=True
    )
    
    # Create dummy labels (random emotions)
    labels = torch.randint(0, 6, (batch_size,))
    
    # Create dummy dialogue context
    dialogue_contexts = []
    for i in range(batch_size):
        context = {
            'previous_utterances': texts[:2],  # Previous 2 utterances
            'speakers': ['Speaker1', 'Speaker2'] * (batch_size // 2 + 1),
            'emotions': ['neutral', 'happiness'] * (batch_size // 2 + 1)
        }
        dialogue_contexts.append(context)
    
    # Create batch in the expected format
    batch = {
        'text_inputs': {
            'input_ids': text_encoding['input_ids'].to(device),
            'attention_mask': text_encoding['attention_mask'].to(device)
        },
        'audio_inputs': {
            'input_values': audio_inputs['input_values'].to(device),
            'attention_mask': audio_inputs.get('attention_mask', torch.ones_like(audio_inputs['input_values'])).to(device)
        },
        'labels': labels.to(device),
        'dialogue_context': dialogue_contexts,
        'raw_texts': texts,
        'raw_speakers': ['Speaker1'] * batch_size
    }
    
    return batch
