# 🇹🇷 Turkish GPT-2 Model

In this repository I release GPT-2 model, that was trained on various texts for Turkish.

The model is meant to be an entry point for fine-tuning on other texts.

## Training corpora

I used a Turkish corpora that is taken from oscar-corpus.

It was possible to create byte-level BPE with Tokenizers library of Huggingface.

With the Tokenizers library, I created a 52K byte-level BPE vocab based on the training corpora.

After creating the vocab, I could train the GPT-2 for Turkish on two 2080TI over the complete training corpus (five epochs).

Logs during training:
https://tensorboard.dev/experiment/3AWKv8bBTaqcqZP5frtGkw/#scalars

## Model weights

Both PyTorch and Tensorflow compatible weights are available.

| Model                             | Downloads
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------
| `redrussianarmy/gpt2-turkish-cased`   | [`config.json`](https://huggingface.co/redrussianarmy/gpt2-turkish-cased/resolve/main/config.json) • [`merges.txt`](https://huggingface.co/redrussianarmy/gpt2-turkish-cased/resolve/main/merges.txt) • [`pytorch_model.bin`](https://huggingface.co/redrussianarmy/gpt2-turkish-cased/resolve/main/pytorch_model.bin) • [`special_tokens_map.json`](https://huggingface.co/redrussianarmy/gpt2-turkish-cased/resolve/main/special_tokens_map.json) • [`tf_model.h5`](https://huggingface.co/redrussianarmy/gpt2-turkish-cased/resolve/main/tf_model.h5) • [`tokenizer_config.json`](https://huggingface.co/redrussianarmy/gpt2-turkish-cased/resolve/main/tokenizer_config.json) • [`traning_args.bin`](https://huggingface.co/redrussianarmy/gpt2-turkish-cased/resolve/main/training_args.bin) • [`vocab.json`](https://huggingface.co/redrussianarmy/gpt2-turkish-cased/resolve/main/vocab.json)

## Using the model

The model itself can be used in this way:

``` python
from transformers import AutoTokenizer, AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained("redrussianarmy/gpt2-turkish-cased")
model = AutoModelWithLMHead.from_pretrained("redrussianarmy/gpt2-turkish-cased")
```

Here's an example that shows how to use the great Transformers Pipelines for generating text:

``` python
from transformers import pipeline
pipe = pipeline('text-generation', model="redrussianarmy/gpt2-turkish-cased",
                 tokenizer="redrussianarmy/gpt2-turkish-cased", config={'max_length':800})   
text = pipe("Akşamüstü yolda ilerlerken, ")[0]["generated_text"]
print(text)
```

### How to clone the model repo?
```
git lfs install
git clone https://huggingface.co/redrussianarmy/gpt2-turkish-cased
```

## Contact (Bugs, Feedback, Contribution and more)
For questions about the GPT2-Turkish model, just open an issue [here](https://github.com/redrussianarmy/gpt2-turkish/issues) 🤗
