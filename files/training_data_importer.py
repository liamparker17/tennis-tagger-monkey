"""
Training Data Importer - Load human-tagged CSVs as ground truth

This module enables training from existing human-tagged CSV files to replace human taggers.

Workflow:
1. User selects CSV file (human annotations)
2. User links to corresponding video file
3. System extracts FVD from video (or loads existing)
4. System maps CSV timestamps -> FVD frames
5. Creates training pairs: (FVD features, human labels)
6. Stores in data/training_data/ for batch training

The CSV format expected is the Dartfish export format with columns like:
- Position: Timestamp in microseconds
- Duration: Point duration
- A2: Serve Data: 1st Serve Ace, 2nd Serve Made, Double Fault, etc.
- A3: Serve Placement: Wide, T, Body
- B2: Return Data: Backhand Return Made, Forehand Return Error, etc.
- E1: Last Shot: Forehand, Backhand, Dropshot, etc.
- E2: Last Shot Winner: Player name if winner
- E3: Last Shot Error: Error type (Down Line Long, etc.)
- XY columns: Shot coordinates in x;y format
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import shutil

logger = logging.getLogger('TrainingDataImporter')


class TrainingDataImporter:
    """
    Import human-tagged CSV files and link them to video FVD data for training.

    This creates training pairs that can be used to train models to match
    human-quality tagging without needing actual human taggers.
    """

    # Column mappings for the Dartfish CSV format
    COLUMN_MAPPINGS = {
        # Serve classification
        'serve_data': 'A2: Serve Data',
        'serve_placement': 'A3: Serve Placement',

        # Return classification
        'return_data': 'B2: Return Data',
        'return_placement': 'B3: Return Placement',
        'return_type': 'B7: Return Type',

        # Serve +1 (server's next shot)
        'srv1_stroke': 'C1: Serve +1 Stroke',
        'srv1_data': 'C2: Serve +1 Data',
        'srv1_placement': 'C3: Serve +1 Placement',

        # Return +1 (returner's next shot)
        'ret1_stroke': 'D1: Return +1 Stroke',
        'ret1_data': 'D2: Return +1 Data',
        'ret1_placement': 'D3: Return +1 Placement',

        # Last shot in rally
        'last_shot': 'E1: Last Shot',
        'last_shot_winner': 'E2: Last Shot Winner',
        'last_shot_error': 'E3: Last Shot Error',
        'last_shot_placement': 'E4: Last Shot Placement',

        # Point outcome
        'point_won': 'F1: Point Won',
        'point_score': 'F2: Point Score',

        # Metadata
        'server': 'A1: Server',
        'returner': 'B1: Returner',
        'stroke_count': 'H1: Stroke Count',

        # Coordinates (x;y format)
        'xy_last_shot': 'XY Last Shot',
        'xy_return': 'XY Return',
        'xy_srv1': 'XY Srv+1',
        'xy_ret1': 'XY Ret+1',
    }

    # Label mappings for training
    SERVE_LABELS = {
        '1st Serve Ace': 'ace',
        '1st Serve Made': 'made',
        '2nd Serve Made': 'made',
        '1st Serve Unreturnable': 'unreturnable',
        'Double Fault': 'fault',
        'Fault Wide': 'fault',
        'Fault Long': 'fault',
        'Fault Net': 'fault',
    }

    STROKE_LABELS = {
        'Forehand': 'forehand',
        'Backhand': 'backhand',
        'Dropshot': 'dropshot',
        'Lob': 'lob',
        'Volley': 'volley',
        'Approach': 'approach',
        'Slice': 'slice',
        'Overhead': 'overhead',
        'Serve': 'serve',
    }

    PLACEMENT_LABELS = {
        'Wide': 'wide',
        'T': 't',
        'Body': 'body',
        'Crosscourt Deep': 'crosscourt_deep',
        'Crosscourt Short': 'crosscourt_short',
        'Down Line Deep': 'down_line_deep',
        'Down Line Short': 'down_line_short',
        'Middle Deep': 'middle_deep',
        'Middle Short': 'middle_short',
        'Inside Out Deep': 'inside_out_deep',
        'Inside Out Short': 'inside_out_short',
        'Inside In Deep': 'inside_in_deep',
        'Inside In Short': 'inside_in_short',
    }

    def __init__(self, training_dir: str = "data/training_data"):
        """
        Initialize Training Data Importer.

        Args:
            training_dir: Directory to store imported training data
        """
        self.training_dir = Path(training_dir)
        self.training_dir.mkdir(parents=True, exist_ok=True)

        # Try to import FVD manager
        self.fvd_manager = None
        try:
            from frame_vector_data import create_fvd_manager
            self.fvd_manager = create_fvd_manager()
        except ImportError:
            logger.warning("FVD manager not available")

    def import_csv(self, csv_path: str, video_path: str,
                   dataset_name: str = None) -> Dict:
        """
        Import human-tagged CSV and link to video FVD.

        Args:
            csv_path: Path to human annotation CSV
            video_path: Path to corresponding video file
            dataset_name: Optional name for the dataset

        Returns:
            Dictionary with import results
        """
        csv_path = Path(csv_path)
        video_path = Path(video_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Generate dataset name if not provided
        if dataset_name is None:
            dataset_name = csv_path.stem.replace(' ', '_')

        # Create dataset directory
        dataset_dir = self.training_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Importing: {csv_path.name} -> {dataset_name}")

        # Parse CSV
        annotations = self.parse_annotation_csv(str(csv_path))
        logger.info(f"Parsed {len(annotations)} points from CSV")

        # Load or extract FVD
        fvd = self._get_or_extract_fvd(str(video_path))

        # Create training pairs
        training_pairs = self.create_training_pairs(fvd, annotations)
        logger.info(f"Created {len(training_pairs)} training pairs")

        # Calculate feature statistics
        fvd_frame_count = len(fvd.get('frames', {})) if fvd else 0
        pairs_with_features = sum(1 for p in training_pairs if p.get('features') and p['features'].get('frame_count', 0) > 0)

        # Save everything
        metadata = {
            'csv_path': str(csv_path.absolute()),
            'video_path': str(video_path.absolute()),
            'dataset_name': dataset_name,
            'import_date': datetime.now().isoformat(),
            'total_points': len(annotations),
            'training_pairs': len(training_pairs),
            'pairs_with_features': pairs_with_features,
            'fps': fvd.get('fps', 30) if fvd else 30,
            'total_frames': fvd.get('total_frames', 0) if fvd else 0,
            'fvd_frame_count': fvd_frame_count,
            'has_fvd': fvd is not None and fvd_frame_count > 0,
        }

        logger.info(f"Training pairs with FVD features: {pairs_with_features}/{len(training_pairs)}")

        # Save metadata
        with open(dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save annotations
        with open(dataset_dir / "annotations.json", 'w') as f:
            json.dump(annotations, f, indent=2)

        # Save training pairs as NPZ
        self._save_training_pairs(training_pairs, dataset_dir)

        # Save label mappings
        label_mapping = self._create_label_mapping(annotations)
        with open(dataset_dir / "label_mapping.json", 'w') as f:
            json.dump(label_mapping, f, indent=2)

        logger.info(f"Dataset saved to: {dataset_dir}")

        return {
            'dataset_name': dataset_name,
            'path': str(dataset_dir),
            'total_points': len(annotations),
            'training_pairs': len(training_pairs),
            'metadata': metadata
        }

    def parse_annotation_csv(self, csv_path: str) -> List[Dict]:
        """
        Parse the human annotation CSV format.

        Args:
            csv_path: Path to CSV file

        Returns:
            List of annotation dictionaries
        """
        # Read CSV with error handling for encoding
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin-1')

        annotations = []

        for idx, row in df.iterrows():
            # Skip header/metadata rows
            name = str(row.get('Name', ''))
            if 'SAVE' in name.upper() or 'DELETE' in name.upper():
                continue

            # Extract timestamp (Position column - in microseconds)
            position = row.get('Position', 0)
            try:
                position_us = int(position) if pd.notna(position) else 0
            except (ValueError, TypeError):
                continue

            # Skip invalid positions
            if position_us <= 0:
                continue

            # Extract duration
            duration = row.get('Duration', 0)
            try:
                duration_us = int(duration) if pd.notna(duration) else 0
            except (ValueError, TypeError):
                duration_us = 0

            # Build annotation
            annotation = {
                'name': name,
                'position_us': position_us,
                'position_ms': position_us / 1000,
                'duration_us': duration_us,
                'duration_ms': duration_us / 1000,
                'row_index': idx,
            }

            # Extract serve data
            serve_data = self._get_cell(row, 'A2: Serve Data')
            if serve_data:
                annotation['serve_data'] = serve_data
                annotation['serve_label'] = self.SERVE_LABELS.get(serve_data, 'made')

            serve_placement = self._get_cell(row, 'A3: Serve Placement')
            if serve_placement:
                annotation['serve_placement'] = serve_placement
                annotation['serve_placement_label'] = self.PLACEMENT_LABELS.get(
                    serve_placement, serve_placement.lower().replace(' ', '_')
                )

            # Extract server
            server = self._get_cell(row, 'A1: Server')
            if server:
                annotation['server'] = server

            # Extract return data
            return_data = self._get_cell(row, 'B2: Return Data')
            if return_data:
                annotation['return_data'] = return_data
                # Parse return stroke type
                if 'Forehand' in return_data:
                    annotation['return_stroke'] = 'forehand'
                elif 'Backhand' in return_data:
                    annotation['return_stroke'] = 'backhand'

                # Parse return outcome
                if 'Winner' in return_data:
                    annotation['return_outcome'] = 'winner'
                elif 'Error' in return_data:
                    annotation['return_outcome'] = 'error'
                elif 'Made' in return_data:
                    annotation['return_outcome'] = 'made'

            return_placement = self._get_cell(row, 'B3: Return Placement')
            if return_placement:
                annotation['return_placement'] = return_placement

            # Extract last shot
            last_shot = self._get_cell(row, 'E1: Last Shot')
            if last_shot:
                annotation['last_shot'] = last_shot
                annotation['last_shot_label'] = self.STROKE_LABELS.get(last_shot, 'forehand')

            # Extract winner/error
            winner = self._get_cell(row, 'E2: Last Shot Winner')
            if winner:
                annotation['winner'] = winner
                annotation['point_result'] = 'winner'

            error = self._get_cell(row, 'E3: Last Shot Error')
            if error:
                annotation['error'] = error
                annotation['point_result'] = 'error'

            # Extract last shot placement
            last_placement = self._get_cell(row, 'E4: Last Shot Placement')
            if last_placement:
                annotation['last_shot_placement'] = last_placement

            # Extract point won
            point_won = self._get_cell(row, 'F1: Point Won')
            if point_won:
                annotation['point_won'] = point_won

            # Extract stroke count
            stroke_count = self._get_cell(row, 'H1: Stroke Count')
            if stroke_count:
                try:
                    annotation['stroke_count'] = int(stroke_count.split()[0]) if ' ' in stroke_count else int(stroke_count)
                except (ValueError, IndexError):
                    pass

            # Extract XY coordinates (format: "81;78")
            for xy_col in ['XY Last Shot', 'XY Return', 'XY Srv+1', 'XY Ret+1']:
                xy_val = self._get_cell(row, xy_col)
                if xy_val and ';' in str(xy_val):
                    try:
                        x, y = str(xy_val).split(';')
                        col_key = xy_col.lower().replace(' ', '_').replace('+', '')
                        annotation[col_key] = {'x': int(x), 'y': int(y)}
                    except (ValueError, IndexError):
                        pass

            annotations.append(annotation)

        return annotations

    def _get_cell(self, row: pd.Series, column: str) -> Optional[str]:
        """Get cell value, returning None for empty/NaN."""
        if column not in row.index:
            return None
        val = row[column]
        if pd.isna(val) or val == '' or str(val).strip() == '':
            return None
        return str(val).strip()

    def timestamp_to_frame(self, position_us: int, fps: float) -> int:
        """
        Convert CSV position (microseconds) to video frame number.

        Args:
            position_us: Position in microseconds from CSV
            fps: Video frames per second

        Returns:
            Frame number
        """
        seconds = position_us / 1_000_000
        frame = int(seconds * fps)
        return max(0, frame)

    def get_frame_range_for_point(self, position_us: int, duration_us: int,
                                   fps: float, padding_frames: int = 30) -> Tuple[int, int]:
        """
        Get frame range for a point (with padding).

        Args:
            position_us: Start position in microseconds
            duration_us: Duration in microseconds
            fps: Video FPS
            padding_frames: Extra frames before/after

        Returns:
            (start_frame, end_frame)
        """
        start_frame = self.timestamp_to_frame(position_us, fps)

        if duration_us > 0:
            end_frame = self.timestamp_to_frame(position_us + duration_us, fps)
        else:
            # Default to 5 seconds if no duration
            end_frame = start_frame + int(5 * fps)

        # Add padding
        start_frame = max(0, start_frame - padding_frames)
        end_frame = end_frame + padding_frames

        return start_frame, end_frame

    def _get_or_extract_fvd(self, video_path: str, auto_extract: bool = True) -> Optional[Dict]:
        """
        Get existing FVD or extract if needed.

        Args:
            video_path: Path to video file
            auto_extract: If True, automatically process video to extract FVD

        Returns:
            FVD dictionary or None
        """
        if self.fvd_manager is None:
            logger.warning("FVD manager not available, returning None")
            return None

        # Try to load existing FVD
        fvd = self.fvd_manager.load_fvd(video_path)

        if fvd is not None:
            logger.info(f"Loaded existing FVD with {len(fvd.get('frames', {}))} frames")
            return fvd

        # FVD doesn't exist - need to process video
        if not auto_extract:
            logger.warning(f"No FVD found for {video_path}. Process video first.")
            return None

        logger.info(f"No FVD found - processing video to extract FVD...")

        # Try to extract FVD by processing the video
        try:
            fvd = self._extract_fvd_from_video(video_path)
            return fvd
        except Exception as e:
            logger.error(f"Failed to extract FVD: {e}")
            raise RuntimeError(
                f"No FVD found and automatic extraction failed: {e}\n\n"
                f"Please process the video first using the 'Tag Video' tab, "
                f"then come back and import your CSV."
            )

    def _extract_fvd_from_video(self, video_path: str) -> Dict:
        """
        Process video to extract FVD (Frame Vector Data).

        This runs the detection pipeline on the video to extract:
        - Player bounding boxes and tracking IDs
        - Ball positions
        - Pose keypoints (if enabled)

        Args:
            video_path: Path to video file

        Returns:
            Extracted FVD dictionary
        """
        import sys
        import traceback
        from pathlib import Path

        # Add src to path for imports
        src_path = Path(__file__).parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        # Step 1: Try to import required modules
        try:
            from main import TennisTagger, setup_logging
            import yaml
        except ImportError as e:
            # Check what's missing
            main_path = Path(__file__).parent / "main.py"
            src_main_path = Path(__file__).parent / "src" / "main.py"

            if not main_path.exists() and not src_main_path.exists():
                raise RuntimeError(f"main.py not found. Expected at:\n  {main_path}\n  or {src_main_path}")

            raise RuntimeError(
                f"Failed to import video processing modules.\n"
                f"Import error: {e}\n\n"
                f"This usually means a dependency is missing.\n"
                f"Try: pip install ultralytics opencv-python torch"
            )

        # Step 2: Load config
        config_path = Path(__file__).parent / "config" / "config.yaml"
        if not config_path.exists():
            config_path = Path(__file__).parent / "config.yaml"

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load config from {config_path}: {e}")
        else:
            logger.warning(f"No config file found, using defaults")
            config = {
                'hardware': {'use_gpu': True, 'use_half_precision': True},
                'models': {
                    'player_detection': {'path': 'yolov8x.pt', 'conf_threshold': 0.5},
                    'ball_detection': {'path': 'models/ball_detector.pt', 'conf_threshold': 0.3},
                },
                'processing': {'default_batch_size': 32},
                'output': {'log_dir': 'logs', 'log_level': 'INFO'}
            }

        logger.info(f"Processing video to extract FVD: {Path(video_path).name}")

        # Step 3: Setup output
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_csv = output_dir / f"{Path(video_path).stem}.csv"

        # Step 4: Create tagger and process
        try:
            processing_logger = setup_logging(config)
            tagger = TennisTagger(config, processing_logger)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize TennisTagger.\n"
                f"Error: {e}\n\n"
                f"This usually means YOLO models are missing.\n"
                f"Check that yolov8x.pt exists or will be auto-downloaded."
            )

        # Step 5: Process video
        try:
            stats = tagger.process_video(
                str(video_path),
                str(temp_csv),
                visualize=False,
                batch_size=32,
                checkpoint_interval=1000,
                resume=True,
                enable_pose=False,
                extract_fvd=True
            )
            logger.info(f"Video processing complete: {stats.get('frames_processed', 0)} frames")
        except Exception as e:
            raise RuntimeError(
                f"Video processing failed.\n"
                f"Error: {e}\n\n"
                f"Traceback:\n{traceback.format_exc()}"
            )

        # Step 6: Load the FVD that was just created
        fvd = self.fvd_manager.load_fvd(video_path)

        if fvd is None:
            # Check if FVD file exists
            fvd_path = self.fvd_manager.get_fvd_path_compressed(video_path)
            fvd_path2 = self.fvd_manager.get_fvd_path(video_path)
            raise RuntimeError(
                f"FVD extraction completed but file not found.\n"
                f"Expected at: {fvd_path} or {fvd_path2}\n"
                f"Processing stats: {stats}"
            )

        logger.info(f"FVD extracted: {len(fvd.get('frames', {}))} frames")
        return fvd

    def create_training_pairs(self, fvd: Optional[Dict],
                              annotations: List[Dict]) -> List[Dict]:
        """
        Create (features, labels) pairs for training.

        Args:
            fvd: FVD dictionary (or None if not available)
            annotations: List of parsed annotations

        Returns:
            List of training pair dictionaries
        """
        training_pairs = []

        fps = fvd.get('fps', 30) if fvd else 30
        frames = fvd.get('frames', {}) if fvd else {}

        for ann in annotations:
            position_us = ann.get('position_us', 0)
            duration_us = ann.get('duration_us', 0)

            if position_us <= 0:
                continue

            # Get frame range
            start_frame, end_frame = self.get_frame_range_for_point(
                position_us, duration_us, fps
            )

            # Create training pair
            pair = {
                'annotation': ann,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'fps': fps,
                'features': None,
                'labels': {}
            }

            # Extract features from FVD frames
            if frames:
                pair['features'] = self._extract_features_from_fvd(
                    frames, start_frame, end_frame
                )

            # Create labels from annotation
            pair['labels'] = self._create_labels(ann)

            training_pairs.append(pair)

        return training_pairs

    def _extract_features_from_fvd(self, frames: Dict,
                                    start_frame: int, end_frame: int) -> Dict:
        """
        Extract features from FVD frames for training.

        Args:
            frames: FVD frames dictionary
            start_frame: Start frame index
            end_frame: End frame index

        Returns:
            Feature dictionary
        """
        features = {
            'player_tracks': [],
            'ball_positions': [],
            'poses': [],
            'frame_count': 0
        }

        for frame_idx in range(start_frame, end_frame + 1):
            frame_key = str(frame_idx)
            if frame_key not in frames:
                continue

            frame_data = frames[frame_key]
            features['frame_count'] += 1

            # Extract player data
            players = frame_data.get('players', [])
            for player in players:
                features['player_tracks'].append({
                    'frame': frame_idx,
                    'bbox': player.get('bbox', []),
                    'id': player.get('id', -1),
                    'conf': player.get('conf', 0),
                    'pose': player.get('pose', [])
                })

            # Extract ball data
            ball = frame_data.get('ball')
            if ball:
                features['ball_positions'].append({
                    'frame': frame_idx,
                    'x': ball.get('x', 0),
                    'y': ball.get('y', 0),
                    'conf': ball.get('c', 0)
                })

        return features

    def _create_labels(self, annotation: Dict) -> Dict:
        """
        Create label dictionary from annotation.

        Args:
            annotation: Annotation dictionary

        Returns:
            Labels dictionary for training
        """
        labels = {}

        # Serve classification labels
        if 'serve_label' in annotation:
            labels['serve_class'] = annotation['serve_label']

        if 'serve_placement_label' in annotation:
            labels['serve_placement'] = annotation['serve_placement_label']

        # Stroke classification labels
        if 'last_shot_label' in annotation:
            labels['stroke_class'] = annotation['last_shot_label']

        if 'return_stroke' in annotation:
            labels['return_stroke'] = annotation['return_stroke']

        # Outcome labels
        if 'point_result' in annotation:
            labels['point_result'] = annotation['point_result']

        if 'return_outcome' in annotation:
            labels['return_outcome'] = annotation['return_outcome']

        # Shot placement labels
        if 'last_shot_placement' in annotation:
            placement = annotation['last_shot_placement']
            labels['shot_placement'] = self.PLACEMENT_LABELS.get(
                placement, placement.lower().replace(' ', '_') if placement else 'unknown'
            )

        return labels

    def _create_label_mapping(self, annotations: List[Dict]) -> Dict:
        """
        Create label to index mapping for all label types.

        Args:
            annotations: List of annotations

        Returns:
            Dictionary with label mappings
        """
        mappings = {
            'serve_class': {},
            'serve_placement': {},
            'stroke_class': {},
            'shot_placement': {},
            'point_result': {},
        }

        # Collect unique labels
        for ann in annotations:
            if 'serve_label' in ann:
                label = ann['serve_label']
                if label not in mappings['serve_class']:
                    mappings['serve_class'][label] = len(mappings['serve_class'])

            if 'serve_placement_label' in ann:
                label = ann['serve_placement_label']
                if label not in mappings['serve_placement']:
                    mappings['serve_placement'][label] = len(mappings['serve_placement'])

            if 'last_shot_label' in ann:
                label = ann['last_shot_label']
                if label not in mappings['stroke_class']:
                    mappings['stroke_class'][label] = len(mappings['stroke_class'])

            if 'point_result' in ann:
                label = ann['point_result']
                if label not in mappings['point_result']:
                    mappings['point_result'][label] = len(mappings['point_result'])

        # Add index to label reverse mapping
        # Use list() to avoid "dictionary changed size during iteration" error
        reverse_mappings = {}
        for category in list(mappings.keys()):
            reverse_mappings[f'{category}_reverse'] = {v: k for k, v in mappings[category].items()}
        mappings.update(reverse_mappings)

        return mappings

    def _save_training_pairs(self, training_pairs: List[Dict],
                              dataset_dir: Path) -> None:
        """
        Save training pairs as NPZ files with BOTH features and labels.

        Args:
            training_pairs: List of training pairs
            dataset_dir: Directory to save to
        """
        # Collect labels
        all_labels = {
            'serve_class': [],
            'serve_placement': [],
            'stroke_class': [],
            'shot_placement': [],
            'point_result': [],
            'frame_indices': [],
        }

        # Collect features - these are what the model will actually learn from
        all_features = {
            'player_bbox_sequences': [],  # List of player bbox sequences per point
            'ball_position_sequences': [],  # List of ball position sequences per point
            'frame_counts': [],  # Number of frames per point
        }

        for i, pair in enumerate(training_pairs):
            labels = pair.get('labels', {})
            features = pair.get('features', {}) or {}

            # Save labels
            all_labels['serve_class'].append(labels.get('serve_class', ''))
            all_labels['serve_placement'].append(labels.get('serve_placement', ''))
            all_labels['stroke_class'].append(labels.get('stroke_class', ''))
            all_labels['shot_placement'].append(labels.get('shot_placement', ''))
            all_labels['point_result'].append(labels.get('point_result', ''))
            all_labels['frame_indices'].append([pair['start_frame'], pair['end_frame']])

            # Save features - extract key info from FVD data
            player_tracks = features.get('player_tracks', [])
            ball_positions = features.get('ball_positions', [])
            frame_count = features.get('frame_count', 0)

            # Convert player tracks to feature vectors
            # Each track: {'frame': int, 'bbox': [x1,y1,x2,y2], 'id': int, 'conf': float, 'pose': [...]}
            player_bboxes = []
            for track in player_tracks:
                bbox = track.get('bbox', [0, 0, 0, 0])
                if len(bbox) >= 4:
                    # Normalize bbox to relative coords and add frame info
                    player_bboxes.append([
                        track.get('frame', 0),
                        track.get('id', 0),
                        bbox[0], bbox[1], bbox[2], bbox[3],
                        track.get('conf', 0)
                    ])

            # Convert ball positions to feature vectors
            ball_pos = []
            for bp in ball_positions:
                ball_pos.append([
                    bp.get('frame', 0),
                    bp.get('x', 0),
                    bp.get('y', 0),
                    bp.get('conf', 0)
                ])

            all_features['player_bbox_sequences'].append(player_bboxes)
            all_features['ball_position_sequences'].append(ball_pos)
            all_features['frame_counts'].append(frame_count)

        # Save labels as NPZ
        np.savez(
            dataset_dir / "training_pairs.npz",
            serve_class=np.array(all_labels['serve_class']),
            serve_placement=np.array(all_labels['serve_placement']),
            stroke_class=np.array(all_labels['stroke_class']),
            shot_placement=np.array(all_labels['shot_placement']),
            point_result=np.array(all_labels['point_result']),
            frame_indices=np.array(all_labels['frame_indices']),
            frame_counts=np.array(all_features['frame_counts']),
        )

        # Save features separately (variable-length sequences)
        # Using JSON for variable-length data
        features_file = dataset_dir / "training_features.json"
        with open(features_file, 'w') as f:
            json.dump({
                'player_bbox_sequences': all_features['player_bbox_sequences'],
                'ball_position_sequences': all_features['ball_position_sequences'],
            }, f)

        logger.info(f"Saved {len(training_pairs)} training pairs to NPZ + features JSON")
        logger.info(f"Feature stats: {sum(all_features['frame_counts'])} total frames across all points")

    def list_imported_datasets(self) -> List[Dict]:
        """
        List all imported training datasets.

        Returns:
            List of dataset metadata dictionaries
        """
        datasets = []

        for dataset_dir in self.training_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            metadata_file = dataset_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Add directory info
                metadata['directory'] = str(dataset_dir)
                metadata['name'] = dataset_dir.name

                # Check if training pairs exist
                npz_file = dataset_dir / "training_pairs.npz"
                metadata['has_training_data'] = npz_file.exists()

                # Get size
                total_size = sum(f.stat().st_size for f in dataset_dir.glob("*"))
                metadata['size_mb'] = round(total_size / (1024 * 1024), 2)

                datasets.append(metadata)

            except Exception as e:
                logger.warning(f"Error reading dataset {dataset_dir}: {e}")

        return sorted(datasets, key=lambda x: x.get('import_date', ''), reverse=True)

    def load_training_data(self, dataset_name: str) -> Tuple[Dict, Dict]:
        """
        Load training data from an imported dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            (features, labels) tuple
        """
        dataset_dir = self.training_dir / dataset_name

        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_name}")

        # Load training pairs
        npz_file = dataset_dir / "training_pairs.npz"
        if not npz_file.exists():
            raise FileNotFoundError(f"Training data not found: {npz_file}")

        data = np.load(npz_file, allow_pickle=True)

        labels = {
            'serve_class': data['serve_class'],
            'serve_placement': data['serve_placement'],
            'stroke_class': data['stroke_class'],
            'shot_placement': data['shot_placement'],
            'point_result': data['point_result'],
            'frame_indices': data['frame_indices'],
        }

        # Load label mapping
        mapping_file = dataset_dir / "label_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                label_mapping = json.load(f)
        else:
            label_mapping = {}

        return labels, label_mapping

    def delete_dataset(self, dataset_name: str) -> bool:
        """
        Delete an imported dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            True if deleted
        """
        dataset_dir = self.training_dir / dataset_name

        if not dataset_dir.exists():
            return False

        try:
            shutil.rmtree(dataset_dir)
            logger.info(f"Deleted dataset: {dataset_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting dataset: {e}")
            return False


# Convenience function
def create_training_data_importer(training_dir: str = "data/training_data") -> TrainingDataImporter:
    """
    Create training data importer instance.

    Args:
        training_dir: Directory for training data

    Returns:
        TrainingDataImporter instance
    """
    return TrainingDataImporter(training_dir)
