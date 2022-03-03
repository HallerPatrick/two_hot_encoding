from flair.data import Sentence
import wrapt

import flair
import torch

from model import RNNModel

_old_load_language_model = flair.models.LanguageModel.load_language_model

@wrapt.patch_function_wrapper(flair.models.LanguageModel, "load_language_model")
def load_language_model(wrapped, instance, args, kwargs):
    """Monkey patch load_language_model to load our RNNModel"""

    state = torch.load(str(args[0]), map_location=flair.device)

    model = RNNModel(
        dictionary=state['dictionary'],
        nlayers=state['nlayers'],
        ngrams=state['ngrams'],
        hidden_size=state['hidden_size'],
        unk_t=state['unk_t'],
        nout=state['nout'],
        embedding_size=state['embedding_size'],
        is_forward_lm=state['is_forward_lm'],
        document_delimiter=state['document_delimiter'],
        dropout=state['dropout'],
    )
    model.load_state_dict(state["state_dict"])
    model.eval()
    model.to(flair.device)

    return model
    

from flair.datasets import NER_ENGLISH_PERSON
from flair.embeddings import FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# 1. get the corpus
corpus = NER_ENGLISH_PERSON()

# TODO: Downsample
print(corpus)

# 2. what label do we want to predict?
label_type = 'ner'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

# 4. initialize embedding stack with Flair and GloVe
embedding_types = [
    # FlairEmbeddings('news-forward'),
    FlairEmbeddings('flair_model.pt'),
]

embeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type,
                        use_crf=True)

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/sota-ner-flair',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)
