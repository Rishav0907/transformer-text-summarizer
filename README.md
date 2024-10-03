# Transformer-based Text Summarization

This project implements a Transformer-based model for text summarization using PyTorch. It's designed to train on the CNN/DailyMail dataset to generate concise summaries of news articles.

## Project Structure

The project is organized into several modules:

- `Main/`: Contains the main training script
- `Bloc/`: Includes the main Transformer model
- `Modules/`: Contains individual components of the Transformer
- `subModules/`: Contains higher-level modules built from basic components
- `configs/`: Holds configuration settings

## Key Components

1. Transformer Model:
   - Encoder-Decoder architecture
   - Multi-head self-attention and cross-attention mechanisms
   - Positional encoding

2. Training:
   - Uses CNN/DailyMail dataset
   - Implements teacher forcing
   - Uses Adam optimizer and CrossEntropyLoss

3. Evaluation:
   - Implements a simple ROUGE-1 score for evaluation

## Configuration

The main configuration is stored in `configs/config.py`. You can adjust hyperparameters like hidden dimensions, number of layers, learning rate, etc., in this file.

## Usage

To train the model:

`python Main/train.py`


To test the model:

`python Main/train.py test`



## Requirements

The project requires PyTorch and other common data science libraries. A full list of requirements can be found in the `requirements.txt` file.

## Model Architecture

The Transformer model is defined in `Bloc/transformer.py`. It consists of an encoder and a decoder, each with multiple layers of self-attention and feed-forward networks.

## Training Process

The training process is implemented in `Main/train.py`. It includes functionality for checkpointing and resuming training from the latest checkpoint.

## Evaluation

The model is evaluated using a simple ROUGE-1 score implementation found in `Main/train.py`.

## Future Improvements

1. Implement more sophisticated ROUGE metrics
2. Experiment with different model architectures and hyperparameters
3. Implement data parallelism for faster training on multiple GPUs
