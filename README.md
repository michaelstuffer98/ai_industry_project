# ai_industry_project
Multi-label audio file classification
## Preprocessing
Steps and description can be found in `feature_extraction`
## Models
- Music Tagging transformer (Full implementation of transformer)
- Wav2Vec transformer (finetuning model)
- Short-chunk CNN
- RNN
- (Naive predictor for baseline metrics)
## Structure
- configs: configuration files to set parameters for preprocessing and the models
- data: contains preprocesed data and labels
- images: images used in markdown of the notebooks
- lib_util: different utilities that were used
- (models: saved instances of trained models, not uploaded to git)
- Plots: plotted results of preprocessing, data analysis and training
- wav_data: raw data, audio files in `WAV` format
- archive: old, experimental implementations, unused
## Dependencies
use `pip install -r requirements.txt`