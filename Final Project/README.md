# eeg-decoding

## Dataset
The ZuCo 1.0 dataset is available at: https://osf.io/q3zws/ <br />
The ZuCo 2.0 dataset is available at: https://osf.io/2urht/

## Getting Started

### Data Preparation
1. **Download EEG Data**: 
   - Obtain the EEG data from the OSF directory.
   - Store the data in `dataset/zuco1` or `dataset/zuco2`.

### Configuration
2. **Set Up for EEG Extraction**:
   - In `config.py`, enable EEG extraction by setting `run_eeg_extraction` to `True`.
   - Specify `feature_set` and `class_task` according to your requirements.
   - Ensure the directory `../eeg_features/` is created to hold the extracted EEG features.

### Running Experiments
3. **Experiment Execution**:
   - For running experiments, set `run_eeg_extraction` to `False`.
   - Adjust other parameters as needed for your specific experiment.
   - Execute `python3 tune_text_model.py` for the text-only baseline model.
   - Use `python3 tune_combi_model.py` to run other models.
   - Use `python3 main.py` to run our EEG decoder

## Notes
- For detailed information on parameter settings and model configurations, refer to the documentation in `config.py`.

## Disclaimer:
- Data collection and preprocessing, as well as most of the starter code (model training + inference, etc.) were done by Nora Hollenstein's team.
- My contribution was the 'Conformer' architecture which combines convolution and self-attention to extract EEG, and implementing the RoBERTa model.
