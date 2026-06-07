# Emotion Recognition Projects

This repository is one emotion-recognition system made of separate replaceable projects/agents.

Each project has its own manifest, feature area, model output, and results folder under:

```text
E:\emotion_recognition_data\agents\
```

## Projects

| Project | Purpose | Workspace |
| --- | --- | --- |
| Audio Agent | Emotion from speech/audio | `E:\emotion_recognition_data\agents\audio` |
| Vision Agent | Emotion from face/images/video frames | `E:\emotion_recognition_data\agents\vision` |
| Text Agent | Emotion from typed text or transcripts | `E:\emotion_recognition_data\agents\text` |
| Animation Agent | Animated-content optimized/not-optimized task | `E:\emotion_recognition_data\agents\animation` |
| Fusion Agent | Late fusion over audio + visual embeddings | `E:\emotion_recognition_data\agents\multimodal` |
| Supervisor Agent | Central agent that compares all available predictions | `E:\emotion_recognition_data\agents\supervisor` |

Initialize or refresh the official workspace:

```powershell
cd E:\emotion_recognition_internship
.\.venv311\Scripts\python.exe src\setup_agent_workspaces.py
```

The generated registry is:

```text
E:\emotion_recognition_data\agents\supervisor\agent_registry.json
```

