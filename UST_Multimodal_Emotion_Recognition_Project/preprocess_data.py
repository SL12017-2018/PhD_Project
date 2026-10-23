"""
IEMOCAP Data Preprocessing Script
Creates train/val/test manifest files from existing IEMOCAP data structure
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
import argparse
import warnings
warnings.filterwarnings('ignore')


class IEMOCAPPreprocessor:
    """Preprocessor for IEMOCAP dataset"""
    
    def __init__(
        self,
        iemocap_root: str,
        output_dir: str,
        emotion_mapping: Optional[Dict] = None
    ):
        self.iemocap_root = Path(iemocap_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard IEMOCAP emotion mapping
        if emotion_mapping is None:
            self.emotion_mapping = {
                'ang': 0,  # angry
                'hap': 1,  # happy  
                'exc': 1,  # excited -> happy (common practice)
                'sad': 2,  # sad
                'neu': 3,  # neutral
                'fru': 4,  # frustrated
                'sur': 5,  # surprise (if exists)
                'fea': 6,  # fear (if exists)
                'dis': 7,  # disgust (if exists)
            }
        else:
            self.emotion_mapping = emotion_mapping
        
        # Reverse mapping for names
        self.emotion_names = {
            0: 'angry', 1: 'happy', 2: 'sad', 3: 'neutral', 
            4: 'frustrated', 5: 'surprise', 6: 'fear', 7: 'disgust'
        }
        
        print(f"IEMOCAP Preprocessor initialized")
        print(f"   Input: {self.iemocap_root}")
        print(f"   Output: {self.output_dir}")
        print(f"   Emotion mapping: {self.emotion_mapping}")
    
    def find_existing_data_structure(self) -> Tuple[Optional[Path], str]:
        """Find existing IEMOCAP data structure"""
        
        # Check different possible structures
        possible_paths = [
            self.iemocap_root,
            self.iemocap_root / "train",
            self.iemocap_root.parent / "iemocap_export" / "train",
            Path("./iemocap_dataset"),
            Path("./iemocap_export"),
        ]
        
        for path in possible_paths:
            if path.exists():
                # Check what type of structure we have
                if (path / "audio").exists() and (path / "metadata").exists():
                    return path, "structured"
                elif any(path.glob("*.csv")):
                    return path, "csv"
                elif any(path.glob("*.json")):
                    return path, "json"
                elif any(path.glob("**/*.wav")):
                    return path, "raw_audio"
        
        return None, "none"
    
    def process_structured_data(self, data_path: Path) -> List[Dict]:
        """Process structured IEMOCAP data (audio + metadata folders)"""
        
        print(f"Processing structured data from {data_path}")
        
        audio_dir = data_path / "audio"
        metadata_dir = data_path / "metadata"
        
        manifest = []
        processed_count = 0
        
        # Find all audio files
        audio_files = list(audio_dir.glob("*.wav"))
        print(f"   Found {len(audio_files)} audio files")
        
        for audio_file in audio_files:
            try:
                # Extract info from filename
                filename = audio_file.stem
                
                # Try different naming patterns
                if '_' in filename:
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        session_info = parts[0]  
                        dialog_info = parts[1]   
                        utt_info = parts[2]      
                    else:
                        # Fallback
                        session_info = filename[:6] if len(filename) > 6 else filename
                        dialog_info = "dialog"
                        utt_info = filename[-4:] if len(filename) > 4 else filename
                else:
                    # Simple fallback
                    session_info = filename[:6] if len(filename) > 6 else filename
                    dialog_info = "dialog"
                    utt_info = filename[-4:] if len(filename) > 4 else filename
                
                # Extract speaker
                speaker = 'F' if 'F' in utt_info else 'M'
                
                # Create dialog and turn IDs
                dialog_id = hash(f"{session_info}_{dialog_info}") % 10000
                turn_id = int(''.join(filter(str.isdigit, utt_info))) if any(c.isdigit() for c in utt_info) else processed_count
                
                # Look for metadata
                metadata_file = metadata_dir / f"{filename}.json"
                text = ""
                emotion = "neu"  # default
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    text = metadata.get('text', '')
                    emotion = metadata.get('emotion', 'neu')
                else:
                    # Try to infer from filename or use defaults
                    text = f"Audio utterance {filename}"
                    
                    # Try to infer emotion from filename patterns
                    filename_lower = filename.lower()
                    if 'ang' in filename_lower:
                        emotion = 'ang'
                    elif 'hap' in filename_lower or 'exc' in filename_lower:
                        emotion = 'hap'
                    elif 'sad' in filename_lower:
                        emotion = 'sad'
                    elif 'fru' in filename_lower:
                        emotion = 'fru'
                    else:
                        emotion = 'neu'
                
                # Map emotion to label
                emotion_key = emotion[:3].lower()  # Take first 3 chars
                if emotion_key in self.emotion_mapping:
                    label = self.emotion_mapping[emotion_key]
                    
                    manifest.append({
                        "dialog_id": dialog_id,
                        "turn_id": turn_id,
                        "speaker": speaker,
                        "text": text,
                        "audio_path": str(audio_file.absolute()),
                        "emotion": emotion,
                        "label": label,
                        "filename": filename
                    })
                    
                    processed_count += 1
                
            except Exception as e:
                print(f"   Warning: Error processing {audio_file}: {e}")
                continue
        
        print(f"Processed {processed_count} samples")
        return manifest
    
    def process_csv_data(self, data_path: Path) -> List[Dict]:
        """Process CSV-based IEMOCAP data"""
        
        print(f"Processing CSV data from {data_path}")
        
        manifest = []
        
        # Find CSV files
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            csv_files = list(data_path.glob("**/*.csv"))
        
        print(f"Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                print(f"   Processing {csv_file.name}: {len(df)} rows")
                
                # Try to identify columns
                text_cols = [col for col in df.columns if 'text' in col.lower() or 'utterance' in col.lower()]
                emotion_cols = [col for col in df.columns if 'emotion' in col.lower() or 'label' in col.lower()]
                speaker_cols = [col for col in df.columns if 'speaker' in col.lower()]
                dialog_cols = [col for col in df.columns if 'dialog' in col.lower() or 'conversation' in col.lower()]
                
                # Use first available columns
                text_col = text_cols[0] if text_cols else 'text'
                emotion_col = emotion_cols[0] if emotion_cols else 'emotion'
                speaker_col = speaker_cols[0] if speaker_cols else None
                dialog_col = dialog_cols[0] if dialog_cols else None
                
                for idx, row in df.iterrows():
                    try:
                        # Extract text
                        if text_col in row and pd.notna(row[text_col]):
                            text = str(row[text_col]).strip()
                        else:
                            text = f"Text utterance {idx}"
                        
                        # Extract emotion
                        if emotion_col in row and pd.notna(row[emotion_col]):
                            emotion = str(row[emotion_col]).lower().strip()
                        else:
                            emotion = 'neutral'
                        
                        # Map emotion
                        emotion_key = emotion[:3] if len(emotion) >= 3 else emotion
                        if emotion_key in self.emotion_mapping:
                            label = self.emotion_mapping[emotion_key]
                        else:
                            # Try full emotion name mapping
                            emotion_full_mapping = {
                                'angry': 0, 'anger': 0,
                                'happy': 1, 'happiness': 1, 'joy': 1, 'excited': 1,
                                'sad': 2, 'sadness': 2,
                                'neutral': 3,
                                'frustrated': 4, 'frustration': 4,
                                'surprise': 5, 'surprised': 5,
                                'fear': 6, 'fearful': 6,
                                'disgust': 7, 'disgusted': 7
                            }
                            label = emotion_full_mapping.get(emotion, 3)  # default to neutral
                        
                        # Extract speaker
                        if speaker_col and speaker_col in row and pd.notna(row[speaker_col]):
                            speaker = str(row[speaker_col]).upper()[:1]
                            if speaker not in ['F', 'M']:
                                speaker = 'F' if idx % 2 == 0 else 'M'  # Alternate
                        else:
                            speaker = 'F' if idx % 2 == 0 else 'M'
                        
                        # Extract dialog info
                        if dialog_col and dialog_col in row and pd.notna(row[dialog_col]):
                            dialog_id = hash(str(row[dialog_col])) % 10000
                        else:
                            dialog_id = idx // 10  # Group every 10 utterances
                        
                        manifest.append({
                            "dialog_id": dialog_id,
                            "turn_id": idx,
                            "speaker": speaker,
                            "text": text,
                            "audio_path": "",  # No audio path from CSV
                            "emotion": emotion,
                            "label": label,
                            "source_file": csv_file.name
                        })
                        
                    except Exception as e:
                        print(f"     Warning: Error processing row {idx}: {e}")
                        continue
                
            except Exception as e:
                print(f"   Warning: Error processing {csv_file}: {e}")
                continue
        
        print(f"Processed {len(manifest)} samples from CSV")
        return manifest
    
    def create_dummy_data(self, num_samples: int = 1000) -> List[Dict]:
        """Create dummy IEMOCAP data for testing"""
        
        print(f"Creating dummy IEMOCAP data ({num_samples} samples)")
        
        # Sample texts for each emotion
        emotion_texts = {
            0: [  # angry
                "I can't believe you did that!", "This is absolutely ridiculous!",
                "I'm so frustrated with this situation!", "That's completely unacceptable!"
            ],
            1: [  # happy
                "This is wonderful news!", "I'm so excited about this!",
                "That's absolutely fantastic!", "I couldn't be happier!"
            ],
            2: [  # sad
                "I feel really down about this.", "This makes me quite sad.",
                "I'm disappointed with how things turned out.", "This is heartbreaking."
            ],
            3: [  # neutral
                "The meeting is scheduled for tomorrow.", "Please review the documents.",
                "I'll send you the information later.", "That's fine with me."
            ],
            4: [  # frustrated
                "This keeps happening over and over.", "I can't seem to get this right.",
                "Why is this so complicated?", "This is getting on my nerves."
            ]
        }
        
        manifest = []
        speakers = ['F', 'M']
        
        for i in range(num_samples):
            # Random emotion
            emotion_label = np.random.choice(list(emotion_texts.keys()))
            
            # Random text for this emotion
            texts = emotion_texts[emotion_label]
            text = np.random.choice(texts)
            
            # Add some variation
            text = f"{text} (Sample {i+1})"
            
            manifest.append({
                "dialog_id": i // 10,  # 10 utterances per dialog
                "turn_id": i % 10,
                "speaker": np.random.choice(speakers),
                "text": text,
                "audio_path": f"dummy_audio_{i:04d}.wav",
                "emotion": self.emotion_names.get(emotion_label, 'neutral'),
                "label": emotion_label,
                "is_dummy": True
            })
        
        print(f"Created {len(manifest)} dummy samples")
        return manifest
    
    def split_data(
        self, 
        manifest: List[Dict], 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_by_emotion: bool = True
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train/val/test sets"""
        
        print(f"Splitting data: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Ratios must sum to 1.0")
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(manifest)
        
        # Stratify by emotion if requested
        if stratify_by_emotion and 'label' in df.columns:
            stratify_col = df['label']
        else:
            stratify_col = None
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            stratify=stratify_col,
            random_state=42
        )
        
        # Second split: val vs test
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        
        if stratify_by_emotion and 'label' in temp_df.columns:
            temp_stratify = temp_df['label']
        else:
            temp_stratify = None
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_test_ratio),
            stratify=temp_stratify,
            random_state=42
        )
        
        # Convert back to list of dicts
        train_manifest = train_df.to_dict('records')
        val_manifest = val_df.to_dict('records')
        test_manifest = test_df.to_dict('records')
        
        print(f" Split complete:")
        print(f"      Train: {len(train_manifest)} samples")
        print(f"      Val:   {len(val_manifest)} samples")
        print(f"      Test:  {len(test_manifest)} samples")
        
        # Print emotion distribution
        for split_name, split_data in [("Train", train_manifest), ("Val", val_manifest), ("Test", test_manifest)]:
            emotion_counts = {}
            for item in split_data:
                emotion = item.get('emotion', 'unknown')
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            print(f"      {split_name} emotions: {emotion_counts}")
        
        return train_manifest, val_manifest, test_manifest
    
    def save_manifests(
        self,
        train_manifest: List[Dict],
        val_manifest: List[Dict],
        test_manifest: List[Dict]
    ):
        """Save manifest files"""
        
        print(f"Saving manifest files to {self.output_dir}")
        
        # Save train manifest
        train_file = self.output_dir / "train_meta.json"
        with open(train_file, 'w') as f:
            json.dump(train_manifest, f, indent=2)
        print(f"Train manifest: {train_file} ({len(train_manifest)} samples)")
        
        # Save val manifest
        val_file = self.output_dir / "val_meta.json"
        with open(val_file, 'w') as f:
            json.dump(val_manifest, f, indent=2)
        print(f"Val manifest: {val_file} ({len(val_manifest)} samples)")
        
        # Save test manifest
        test_file = self.output_dir / "test_meta.json"
        with open(test_file, 'w') as f:
            json.dump(test_manifest, f, indent=2)
        print(f"Test manifest: {test_file} ({len(test_manifest)} samples)")
        
        # Save metadata
        metadata = {
            "emotion_mapping": self.emotion_mapping,
            "emotion_names": self.emotion_names,
            "num_classes": len(set(self.emotion_mapping.values())),
            "total_samples": len(train_manifest) + len(val_manifest) + len(test_manifest),
            "splits": {
                "train": len(train_manifest),
                "val": len(val_manifest),
                "test": len(test_manifest)
            }
        }
        
        metadata_file = self.output_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata: {metadata_file}")
    
    def process_dataset(
        self,
        create_dummy: bool = False,
        dummy_samples: int = 1000,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        """Complete dataset processing pipeline"""
        
        print("Starting IEMOCAP dataset preprocessing")
        print("=" * 50)
        
        if create_dummy:
            # Create dummy data
            manifest = self.create_dummy_data(dummy_samples)
        else:
            # Find and process existing data
            data_path, data_type = self.find_existing_data_structure()
            
            if data_type == "none":
                print("No IEMOCAP data found. Creating dummy data instead.")
                manifest = self.create_dummy_data(1000)
            elif data_type == "structured":
                manifest = self.process_structured_data(data_path)
            elif data_type == "csv":
                manifest = self.process_csv_data(data_path)
            else:
                print(f"Data type '{data_type}' not fully supported. Creating dummy data.")
                manifest = self.create_dummy_data(1000)
        
        if not manifest:
            print("No data to process. Exiting.")
            return
        
        # Split data
        train_manifest, val_manifest, test_manifest = self.split_data(
            manifest, train_ratio, val_ratio, test_ratio
        )
        
        # Save manifests
        self.save_manifests(train_manifest, val_manifest, test_manifest)
        
        print("\nDataset preprocessing completed successfully!")
        print(f"Output directory: {self.output_dir}")
        print(f"Total samples: {len(manifest)}")
        print(f"Emotion classes: {len(set(self.emotion_mapping.values()))}")


def main():
    """Main preprocessing function"""
    
    parser = argparse.ArgumentParser(description='Preprocess IEMOCAP Dataset')
    
    parser.add_argument(
        '--iemocap_root',
        type=str,
        default='../iemocap_dataset',
        help='Path to IEMOCAP root directory'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data',
        help='Output directory for processed data'
    )
    
    parser.add_argument(
        '--create_dummy',
        action='store_true',
        help='Create dummy data instead of processing real data'
    )
    
    parser.add_argument(
        '--dummy_samples',
        type=int,
        default=1000,
        help='Number of dummy samples to create'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Training set ratio'
    )
    
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )
    
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Test set ratio'
    )
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = IEMOCAPPreprocessor(
        iemocap_root=args.iemocap_root,
        output_dir=args.output_dir
    )
    
    # Process dataset
    preprocessor.process_dataset(
        create_dummy=args.create_dummy,
        dummy_samples=args.dummy_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )


if __name__ == "__main__":
    main()
