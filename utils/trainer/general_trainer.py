from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from recbole.trainer import Trainer


class DSERTrainer(Trainer):
    """Custom trainer for DSER model that handles Doc2Vec pre-training"""

    def __init__(self, config, model):
        super().__init__(config, model)
        self.doc2vec_trained = False

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        """Train one epoch for DSER model"""
        # First, check if Doc2Vec needs to be trained
        if not self.doc2vec_trained and hasattr(self.model, 'requires_doc2vec_training'):
            if self.model.requires_doc2vec_training:
                print("Pre-training Doc2Vec models...")
                self._pretrain_doc2vec(train_data.dataset)
                self.doc2vec_trained = True

        # Then proceed with normal training
        return super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)

    def _pretrain_doc2vec(self, dataset):
        """Pre-train Doc2Vec models for users and items"""
        # Extract user-item interactions from dataset
        inter_feat = dataset.inter_feat
        user_id = inter_feat[dataset.uid_field].numpy()
        item_id = inter_feat[dataset.iid_field].numpy()

        # Build user sequences
        user_sequences = {}
        for uid, iid in zip(user_id, item_id):
            if uid not in user_sequences:
                user_sequences[uid] = []
            user_sequences[uid].append(str(iid))

        # Build item sequences
        item_sequences = {}
        for uid, iid in zip(user_id, item_id):
            if iid not in item_sequences:
                item_sequences[iid] = []
            item_sequences[iid].append(str(uid))

        # Train user Doc2Vec model
        print("Training user Doc2Vec model...")
        user_corpus = []
        for uid, sequence in user_sequences.items():
            if len(sequence) > 0:
                tagged_doc = TaggedDocument(words=sequence, tags=[f'user_{uid}'])
                user_corpus.append(tagged_doc)

        self.model.user_doc2vec = Doc2Vec(
            vector_size=self.model.doc2vec_vector_size,
            window=self.model.doc2vec_window,
            min_count=self.model.doc2vec_min_count,
            workers=4,
            dm=self.model.doc2vec_dm,
            epochs=self.model.doc2vec_epochs
        )
        self.model.user_doc2vec.build_vocab(user_corpus)
        self.model.user_doc2vec.train(
            user_corpus,
            total_examples=self.model.user_doc2vec.corpus_count,
            epochs=self.model.user_doc2vec.epochs
        )

        # Train item Doc2Vec model
        print("Training item Doc2Vec model...")
        item_corpus = []
        for iid, sequence in item_sequences.items():
            if len(sequence) > 0:
                tagged_doc = TaggedDocument(words=sequence, tags=[f'item_{iid}'])
                item_corpus.append(tagged_doc)

        self.model.item_doc2vec = Doc2Vec(
            vector_size=self.model.doc2vec_vector_size,
            window=self.model.doc2vec_window,
            min_count=self.model.doc2vec_min_count,
            workers=4,
            dm=self.model.doc2vec_dm,
            epochs=self.model.doc2vec_epochs
        )
        self.model.item_doc2vec.build_vocab(item_corpus)
        self.model.item_doc2vec.train(
            item_corpus,
            total_examples=self.model.item_doc2vec.corpus_count,
            epochs=self.model.item_doc2vec.epochs
        )

        # Extract and store embeddings
        self.model._save_doc2vec_embeddings(dataset)
        self.model.requires_doc2vec_training = False
