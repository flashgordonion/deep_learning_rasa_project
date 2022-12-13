import torch
from torch import nn
from transformers import BertForSequenceClassification


class BasicTransformer(nn.Module):

    def __init__(self, input_size=64, max_length=64, embed_dim=1024, num_labels=8):
        super(BasicTransformer, self).__init__()
        self.num_labels = num_labels

        self.word_encoding = nn.Embedding(30522, embed_dim)
        self.pos_encoding = nn.Embedding(max_length, embed_dim)

        self.transformer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )

        self.classifier = nn.Linear(embed_dim, out_features=num_labels)
        self.softmax = nn.Softmax(dim=1)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs):
        we = self.word_encoding(inputs)
        # TODO - positional_encoding

        outputs = self.transformer(we)
        classes = self.classifier(outputs)
        # TODO - potentially just select first
        selected_classes = classes[:, 0, :]
        probs = self.softmax(selected_classes)
        return probs

    def forward_train(self, inputs, labels):
        probs = self.forward(inputs)
        label_prob = nn.functional.one_hot(labels, num_classes=self.num_labels).to(torch.float32)
        loss = self.ce(probs, label_prob)
        return probs, loss