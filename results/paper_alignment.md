# Paper Alignment: Modular Agentic Multimodal Emotion Recognition

## Reference Direction

This project is now framed around the reference-paper idea that multimodal emotion recognition should be modular, replaceable, and supervised by a coordinating agent rather than built as one fixed monolithic model.

Reference papers:

1. Dabral, Bansal, Maheshwari, Sharma, Singh, and Kumar, "Multimodal Trait and Emotion Recognition via Agentic AI: An End-to-End Pipeline," ACM Multimedia Workshop, 2025. DOI: https://doi.org/10.1145/3746277.3760412
2. Nepomnyaschiy, Pereziabov, Tliamov, Mikhailov, and Afanasyev, "Agent-Based Modular Learning for Multimodal Emotion Recognition in Human-Agent Systems," arXiv:2512.10975. https://arxiv.org/abs/2512.10975

## How Our System Maps to the Papers

| Paper concept | Project implementation |
| --- | --- |
| Modality-specific agents | Audio agent uses Wav2Vec2 embeddings and the current RAVDESS SVM/fine-tuning path. Visual agent uses ViT embeddings. Fusion agent uses late fusion over projected audio and visual embeddings. |
| Central supervisor | `src/agent_pipeline.py` implements `EmotionSupervisorAgent`, which compares candidate predictions from available modalities and selects the final result. |
| Replaceable modules | The audio branch can switch from the current SVM to fine-tuned Wav2Vec2 without rewriting the visual or fusion code. The visual branch can later be replaced with a stronger face model. |
| End-to-end demo pipeline | `streamlit_app.py` runs the deployed interface. It accepts audio, image, and video uploads, extracts signals, predicts emotion, and records feedback. |
| Feedback loop | `outputs/feedback.csv` stores whether the user says the prediction was correct and the corrected label when available. This is the start of the self-improvement loop. |
| Real-world evaluation | RAVDESS uses actor-independent splits so test actors are unseen during training. This gives a more realistic estimate than testing on the same actor distribution. |

## Current Architecture

1. Audio Agent
   - Backbone: Wav2Vec2 embedding extraction.
   - Current deployed classifier: actor-independent RAVDESS audio SVM.
   - Stronger path: `src/train_audio_wav2vec2.py` for direct Wav2Vec2 fine-tuning.

2. Visual Agent
   - Backbone: ViT embedding extraction from still images or video frames.
   - Current limitation: facial-expression accuracy is weaker than audio on the present data.

3. Fusion Agent
   - Projects audio and visual embeddings to 256 dimensions each.
   - Concatenates them into a 512-dimensional feature.
   - Classifier: fully connected layer, LayerNorm, GELU, Dropout, output classes.

4. Supervisor Agent
   - Runs in the demo's recommended mode.
   - Compares audio, visual, and fusion predictions when those signals are available.
   - Applies a small reliability adjustment because raw confidence values from different models are not fully calibrated.

5. Feedback Agent
   - Saves uploaded media references, predicted label, confidence, correctness, and corrected label.
   - Future training can convert confirmed feedback into new labeled examples.

## Honest Gap From the Papers

The project is now paper-focused, but it is not yet a complete reproduction of the papers.

- The ACM paper includes broader agentic AI behavior such as personality/trait inference and dialogue response. Our current system focuses on emotion recognition.
- The arXiv paper discusses vision, audio, and text agents. Our current system supports audio, image, and video-derived audio/frames, but not a transcript/text emotion agent yet.
- The current supervisor is a practical deterministic supervisor, not an LLM reasoning loop.
- Current accuracy is limited by dataset size, acted-emotion domain shift, visual branch weakness, and CPU-only training constraints.

## Next Paper-Focused Improvements

1. Finish Wav2Vec2 fine-tuning on the full RAVDESS manifest, then compare it against the current audio SVM.
2. Add a face-cropping step before ViT so visual predictions use the face region instead of full frames.
3. Aggregate multiple video frames using mean/attention pooling instead of using only a small frame sample.
4. Add a text/transcript agent using Whisper for speech-to-text and a transformer text classifier.
5. Calibrate agent confidence scores on the validation split so the supervisor makes better cross-modal decisions.
6. Turn `outputs/feedback.csv` into a retraining manifest after manual review.

