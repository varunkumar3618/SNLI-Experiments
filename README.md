# SNLI-Experiments
CS 224N Project, Experiments on the Stanford Natural Language Inference Dataset.

# Instructions
Run "pip install --upgrade -r requirements.txt" to install all requirements except Tensorflow. Then, install either the cpu or the gpu version of Tensorflow using "pip install --upgrade tensorflow" or "pip install --upgrade tensorflow-gpu" respectively.

The final step is to download the nltk tokenizer. To do this, open the python terminal and run the following commands:
import nltk
nltk.download("punkt")

# Models

Sum of Words Model
We use the sum of words model described in "A large annotated corpus for learning natural language inference" (Bowman et al.)  as a baseline. You can run this with:
python main.py --word_embed_dim 100 --hidden_size 200
