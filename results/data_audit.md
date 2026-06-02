# Data Audit

Raw directory: `C:\Users\Acer\emotion_recognition_internship\data\raw`
Raw file count: `3117`

## Raw Extensions
- `.wav`: 1738
- `.npy`: 1258
- `.mp4`: 115
- `.zip`: 4
- `.csv`: 2

## Zip Files
- `animated.zip`: 2517 files, extensions {'.wav': 1258, '.npy': 1258, '.csv': 1}
- `fer.zip`: 1 files, extensions {'.csv': 1}
- `ravdess.zip`: 121 files, extensions {'.mp4': 115, '.zip': 6}
- `savee.zip`: 480 files, extensions {'.wav': 480}

## Manifests
- `labels.csv`: 480 rows, labels {'anger': 60, 'disgust': 60, 'fear': 60, 'happiness': 60, 'neutral': 120, 'sadness': 60, 'surprise': 60}, splits {'train': 336, 'val': 72, 'test': 72}, missing audio 0, missing image 0
- `labels_animated.csv`: 1258 rows, labels {'not_optimized': 620, 'optimized': 638}, splits {'train': 880, 'val': 189, 'test': 189}, missing audio 0, missing image 0
- `labels_ravdess.csv`: 115 rows, labels {'anger': 16, 'disgust': 16, 'fear': 16, 'happiness': 16, 'neutral': 19, 'sadness': 16, 'surprise': 16}, splits {'train': 79, 'val': 15, 'test': 21}, missing audio 0, missing image 0
