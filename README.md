# Text Generation with GPT-2

This repository contains the code and resources used to train and fine-tune GPT-2 on Shakespeare's texts for text generation purposes. The project demonstrates data preparation, model training, and fine-tuning techniques applied to GPT-2 for text generation in Shakespeare's style.

## Project Overview

The goal of this project is to train and fine-tune GPT-2 using text data from Shakespeare's works. Additionally, comparisons are made with a Bigram Language Model to highlight differences in model performance.

### Main Steps
1. **Text Preparation**:
   - Creation of a dataset with 70,000 words.
   - Tokenization and encoding of the text data.
   
2. **Model Training**:
   - Fine-tuning GPT-2 using the prepared text dataset.
   - Use of methods to optimize the loss function during training.
   
### Key Components

- **Self-Attention Mechanism**: The self-attention mechanism used in the GPT-2 architecture enables the model to understand relationships between words in a sequence more effectively than simpler models like Bigram LM.
  
- **Transformer Architecture**: GPT-2 leverages the Transformer architecture, enabling more powerful language modeling and text generation.

### Hyperparameters

The hyperparameters were selected to balance training speed and model performance. Key optimizations include the use of dropout to prevent overfitting and careful tuning of the learning rate.

## Results

The initial results show the model's ability to generate coherent text from prompts. Examples of generated text can be found [here](more.txt).

### Improving the Model

Several improvements can be made to enhance the model’s performance:
- **Change Tokenizer**: Experiment with different tokenization strategies.
- **Hyperparameter Tuning**: Use libraries like Weights & Biases (wandb) for further hyperparameter optimization.
- **Prompt Engineering**: Improve content creation by experimenting with different prompts.

## How to Use

1. Clone the repository by running the following command in your terminal:
```
   git clone https://github.com/pavelMerlin/gpt2
```
2. Install dependencies by executing this command:
```
   pip install -r requirements.txt
```
3. Prepare the data and train the model by running:
```
   python train_gpt2.py
```
## Resources

- [Language Models are Unsupervised Multitask Learners](https://github.com/openai/gpt-2)
- [GPT-2: 1.5B release | OpenAI](https://openai.com/research/gpt-2-1-5b-release)
- [Hugging Face - Transformers](https://github.com/huggingface/transformers)
- [How do Transformers work? - Hugging Face NLP Course](https://huggingface.co/course/chapter1/1)

## Conclusion

The project demonstrates the potential of GPT-2 in text generation tasks, and outlines future directions, including improvements to tokenization and fine-tuning strategies.
