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


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER,
        DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR,
    ], is_trainable=True
)
class CustomDietClassifier(DIETClassifier):

    @classmethod
    def create(
            cls,
            config: Dict[Text, Any],
            model_storage: ModelStorage,
            resource: Resource,
            execution_context: ExecutionContext,
    ) -> GraphComponent:
        # TODO: Implement this
        resp = super(CustomDietClassifier, cls).create(config, model_storage, resource, execution_context)
        return resp

    # def train(self, training_data: TrainingData) -> Resource:
    #
    #     resp = super(CustomDietClassifier, self).train(training_data)
    #     return resp

    def process(self, messages: List[Message]) -> List[Message]:

        resp = super(CustomDietClassifier, self).process(messages)
        return resp
