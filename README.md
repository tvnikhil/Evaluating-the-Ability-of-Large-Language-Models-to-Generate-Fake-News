# Project

## For Paper 1

Follow this for paper 1

## Most Important Requirements

- Python >= 3.9
- PyTorch > 1.0 (with NVIDIA CUDA if required)
- Pandas
- Numpy
- Tqdm
- Instructor
- google-cloud-sdk
- vertex-ai
- openai
- pydantic

## Model

The pre-trained model, `parameter_mdfend.pkl`, is saved in the following path:

mdfend-model/param_model/mdfend/parameter_mdfend.pkl

This model was pre-trained using NVIDIA CUDA. NVIDIA-based machines were used for training, while testing was performed on both NVIDIA and MPS (Apple's Metal Performance Shaders) computers. The results between these systems are very similar, with only minor differences in floating-point precision (up to 2–3 decimal places).

The code used for training/testing was based on the author's original code, with slight modifications to enable compatibility with macOS MPS. For the CUDA version, refer to the author's [original code](https://github.com/ICTMCG/M3FEND/blob/main/models/mdfend.py).

## Dataset Paths

Paths for datasets need to be set manually. These paths are machine-specific.

## Temperature Settings for LLM Modifications

The temperature values used for modifications have been selected after extensive testing. The **best** temperature (temp2) is the one at which the LLMs most effectively reduced the accuracy of the MDFEND model when tested on a modified dataset. The **worst** temperature (temp1) is the opposite, where the accuracy is highest.

## Running Training and Testing

1. Navigate to the project directory.
2. Run the following command to initiate training and testing:

```bash
python main.py --gpu 1 --lr 0.0007 --model_name mdfend --dataset en --domain_num 3

```

## Running LLM Modifications

Authenticate to Google Cloud.

Modify the parameters in the gen-lang-client-0539303742-0d4b2719f5ec.json file.

To run the Instructor version of Gemini (for better performance and fewer errors), navigate to the src directory and run main.py.

For other versions, go to the llm-modifications-ipynb directory and run the appropriate Jupyter notebook.

## Outputs will be saved in the terminal-outputs directory

## Benchmarking

The benchmarking code and results are located in the benchmarking-chatgpt-turbo directory. You can directly run each cell in the associated Jupyter notebook after adjusting the data paths (which may differ depending on your computer).

## For Paper 2

Follow this for paper 2

## To train and test

Go to dir: Code

Change dataset paths and run each cell in ipynb file for the required model (to train and test).

## Modified datasets

Modifed datasets are present in ModifiedDataset dir.

## Raw datasets

Raw datasets are present in OldDataset dir.

## Outputs

Outputs for everything is in Outputs directory.

## Code for modifications through LLM

It is in Code/llm-modifications directory.

## Each cell (in every notebook file) can be run with all the requirements installed similar to paper 1
# Evaluating-the-performance-of-LLMs-with-existing-fake-news-detection-models
