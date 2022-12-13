from typing import Dict, Text, Any, List

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.classifiers.diet_classifier import DIETClassifier

import torch
import torch.nn
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

import numpy as np
import time
import datetime
import random
import numpy as np

import os
import pickle

from custom_components.basic_transformer import BasicTransformer


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER,
        DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR,
    ], is_trainable=True
)
class CustomNLUComponent(GraphComponent):

    def __init__(self, component_config):

        self.config = component_config
        self.epochs = component_config['epochs']

        PATH = "./custom_components/trained_models/basic_transformer"
        self.path = PATH
        if os.path.exists(PATH):
            self.tokenizer = BertTokenizer.from_pretrained(PATH, local_files_only=True)
            model = BasicTransformer()
            model.load_state_dict(torch.load(f"{PATH}/model.pickle"))
            model.eval()
            self.model = model
            with open(f"{PATH}/intent_list.pickle", 'rb') as f:
                self.intent_list = pickle.load(f)

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config)

    def train(self, training_data: TrainingData) -> Resource:

        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU.
            device = torch.device("cuda")

            print('There are %d GPU(s) available.' % torch.cuda.device_count())

            print('We will use the GPU:', torch.cuda.get_device_name(0))

        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")

        # Implement this if your component requires training
        sentences, labels, intent_list = CustomNLUComponent._format_training_data(training_data)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        input_ids, attention_masks = CustomNLUComponent._tokenize_training_data(sentences, tokenizer)

        train, val = CustomNLUComponent._train_val_split(input_ids, attention_masks, labels)
        train_dataloader, validation_dataloader = CustomNLUComponent._make_loaders(train, val)

        model = BasicTransformer()
        model.cuda()

        optimizer = AdamW(
            model.parameters(),
            lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
            eps=1e-8  # args.adam_epsilon  - default is 1e-8.
        )

        epochs = self.epochs

        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,  # Default value in run_glue.py
            num_training_steps=total_steps)

        def flat_accuracy(preds, labels):
            pred_flat = np.argmax(preds, axis=1).flatten()
            labels_flat = labels.flatten()
            return np.sum(pred_flat == labels_flat) / len(labels_flat)

        def format_time(elapsed):
            '''
            Takes a time in seconds and returns a string hh:mm:ss
            '''
            # Round to the nearest second.
            elapsed_rounded = int(round((elapsed)))

            # Format as hh:mm:ss
            return str(datetime.timedelta(seconds=elapsed_rounded))

        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # For each epoch...
        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode. Don't be mislead--the call to
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()

                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                logits, loss = model.forward_train(b_input_ids, b_labels)

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_train_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                # Unpack this training batch from our dataloader.
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels

                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here:
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    logits, loss = model.forward_train(b_input_ids, b_labels)

                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += flat_accuracy(logits, label_ids)

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

        CustomNLUComponent._save_model("basic_transformer", model, tokenizer, intent_list, training_stats)


    @staticmethod
    def _save_model(dir, model, tokenizer, intent_list, training_stats):
        output_dir = f'./custom_components/trained_models/{dir}/'

        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        torch.save(model.state_dict(), f"{output_dir}model.pickle")
        tokenizer.save_pretrained(output_dir)

        with open(f"{output_dir}training_stats.pickle", 'wb') as f:
            pickle.dump(training_stats, f)

        with open(f"{output_dir}intent_list.pickle", 'wb') as f:
            pickle.dump(intent_list, f)

    @staticmethod
    def _format_training_data(training_data: TrainingData):

        inputs, labels = [], []
        intent_list = list(training_data.intents)
        nlu_examples = training_data.nlu_examples
        for example in nlu_examples:
            example_data = example.data
            inputs.append(example_data['text'])
            first_intent = example_data['intent_tokens'][0]

            label_name = first_intent.lemma
            labels.append(intent_list.index(label_name))

        labels = torch.tensor(labels)
        return inputs, labels, intent_list

    @staticmethod
    def _tokenize_training_data(sentences, tokenizer):
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in sentences:
            encoded_dict = tokenizer.encode_plus(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=64,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return input_ids, attention_masks

    @staticmethod
    def _train_val_split(input_ids, attention_masks, labels):
        dataset = TensorDataset(input_ids, attention_masks, labels)

        # Create a 90-10 train-validation split.

        # Calculate the number of samples to include in each set.
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(val_size))
        return train_dataset, val_dataset

    @staticmethod
    def _make_loaders(train, val):
        batch_size = 32

        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order.
        train_dataloader = DataLoader(
            train,  # The training samples.
            sampler=RandomSampler(train),  # Select batches randomly
            batch_size=batch_size  # Trains with this batch size.
        )

        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
            val,  # The validation samples.
            sampler=SequentialSampler(val),  # Pull out batches sequentially.
            batch_size=batch_size  # Evaluate with this batch size.
        )
        return train_dataloader, validation_dataloader

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        # TODO: Implement this if your component augments the training data with
        #       tokens or message features which are used by other components
        #       during training.
        ...

        return training_data

    @staticmethod
    def _convert_to_rasa(prediction, confidence):
        intent = {"name": prediction, "confidence": confidence}
        return intent

    def process(self, messages: List[Message]) -> List[Message]:

        for message in messages:
            sent = message.data['text']
            sentences = [sent]
            input_ids, attention_masks = CustomNLUComponent._tokenize_training_data(sentences, self.tokenizer)

            logits = self.model(input_ids)

            prediction = logits.argmax().item()
            confidence = logits[:, prediction].item()
            selected_intent = self.intent_list[prediction]

            intent = CustomNLUComponent._convert_to_rasa(selected_intent, confidence)
            message.set("intent", intent, add_to_output=True)

        return messages
