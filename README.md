# Noise Cancellation using CNN and LSTM

## Overview
This project implements noise cancellation to remove stationary background noises from audio recordings. The system effectively isolates the target audio signal, such as a person's voice, while reducing unwanted noise.

### Approach
1. **Audio Collection**
   - Three types of audio samples were collected:
     - Person's voice (`clearnoise`).
     - Background noise (`bgnoise`).
     - Person's voice with background noise (`cbnoise`).

2. **Short-Time Fourier Transform (STFT)**
   - The STFT was applied to the audio signals to convert them from the time domain to the frequency domain.
   - Short overlapping windows in STFT help capture non-stationary characteristics of the audio.

3. **Log Spectrogram Transformation**
   - Logarithmic transformation was applied to compress the dynamic range of the signal.
   - This transformation reduces the impact of loud noises and enhances subtle frequency patterns for better noise reduction.

4. **CNN-LSTM Model**
   - **Convolutional Neural Network (CNN):** Extracts spatial features from the spectrogram.
   - **Long Short-Term Memory (LSTM):** Captures temporal dependencies (patterns over time).
   - The model was trained on the log spectrograms of the collected audio samples.
   - Achieved an accuracy of 96% in isolating the target signal from the noisy mixture.

## Workflow
1. **Data Preprocessing**
   - Audio signals were loaded and trimmed to ensure uniform length.
   - Magnitude spectrograms were computed using STFT.
   - Logarithmic transformations were applied.

2. **Model Architecture**
   - CNN layers to extract spatial features.
   - MaxPooling layers to reduce dimensionality and computational load.
   - LSTM layers to process temporal sequences.
   - Dense layers to map extracted features to the target output.
   - Final output reshaped to match the dimensions of the input audio spectrogram.

3. **Noise Reduction Algorithm**
   - A mask was created to separate noise from the target signal based on spectral magnitude comparisons.
   - The mask was applied to the combined audio spectrogram to filter out noise.
   - The Inverse STFT (ISTFT) was used to reconstruct the audio signal in the time domain.

4. **Post-Processing**
   - Normalization and smoothing were applied to ensure the audio signal's quality.
   - The final output was exported in both WAV and MP3 formats.

## Results
- Achieved a significant reduction in background noise while preserving the quality of the target audio signal.
- The noise-canceled audio was evaluated visually (using spectrograms) and audibly for quality assurance.

## Files
- `clearnoise.mp3`: Person's voice.
- `bgnoise.mp3`: Background noise.
- `cbnoise.mp3`: Person's voice with background noise.
- `masked.wav`: Noise-canceled output.
- `output3.mp3`: Final processed audio.

## Tools and Libraries
- **Python Libraries:**
  - `librosa`: For audio analysis and processing.
  - `tensorflow.keras`: For building and training the CNN-LSTM model.
  - `numpy`: Numerical computations.
  - `matplotlib`: Visualization of spectrograms and results.
  - `pandas`: Data preprocessing and interpolation.
  - `pydub`: For exporting audio files in different formats.
  - `scikit-learn`: Data normalization and scaling.

## How to Run
1. Install dependencies:
   ```bash
   pip install librosa numpy matplotlib tensorflow scikit-learn pydub pandas
   ```
2. Run the main script to train the model and generate noise-canceled audio.
3. Inspect the results in `output3.mp3`.

## Future Improvements
- Enhance the model's generalization to handle non-stationary noises.
- Test with diverse datasets to improve robustness.
- Explore advanced architectures such as Transformers for better noise reduction.

## Conclusion
This project demonstrates an effective method to isolate audio signals and remove background noise using a combination of signal processing and deep learning techniques. The integration of CNN and LSTM ensures accurate noise cancellation while preserving temporal and spectral details.

