import torch
from torch import nn
from transformers import BertForSequenceClassification


class StudentTransformer(nn.Module):

    def __init__(self, input_size=64, max_length=64, embed_dim=1024, num_labels=8, teacher_weight=0.5, bert=None, device=None):
        super(StudentTransformer, self).__init__()
        self.num_labels = num_labels
        self.teacher_weight = teacher_weight
        self.device = device

        self.bert = bert

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

        # positional_encoding
        batch_size = inputs.shape[0]
        sentence_size = inputs.shape[1]
        pos = [i for i in range(sentence_size)]
        pos_tensor = torch.Tensor(pos).to(torch.int64)
        pos_tensor_list = [pos_tensor for i in range(batch_size)]
        pos_tensors = torch.stack(pos_tensor_list)
        pos_tensors = pos_tensors.to(self.device)
        p_embed = self.pos_encoding(pos_tensors)

        embed = we + p_embed

        outputs = self.transformer(embed)
        classes = self.classifier(outputs)

        selected_classes = classes[:, 0, :]
        probs = self.softmax(selected_classes)
        return probs

    def bert_probs(self, inputs, input_mask, labels):
        res = self.bert(inputs,
                    token_type_ids=None,
                    attention_mask=input_mask,
                    labels=labels)
        logits = res.logits
        bert_probs = self.softmax(logits)
        return bert_probs

    def weigh_probs(self, probs, bert_probs, label_probs):
        b_loss = self.teacher_weight * self.ce(probs, bert_probs)
        l_loss = (1 - self.teacher_weight) * self.ce(probs, label_probs)
        loss = b_loss + l_loss
        return loss

    def forward_train(self, inputs, input_masks, labels):
        probs = self.forward(inputs)
        bert_probs = self.bert_probs(inputs, input_masks, labels)
        label_probs = nn.functional.one_hot(labels, num_classes=self.num_labels).to(torch.float32)

        loss = self.weigh_probs(probs, bert_probs, label_probs)

        return probs, loss
