from __future__ import annotations
from typing import List, Dict, Text, Any, Optional
from xml import dom

from rasa.engine.storage.resource import Resource

from rasa.core.policies.policy import Policy
from rasa.core.policies.ted_policy import TEDPolicy, PolicyPrediction, Domain, MessageContainerForCoreFeaturization
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.trackers import DialogueStateTracker


# Modified test policy from https://zdatainc.com/custom-rasa-policies/

@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT], is_trainable=False
)
class MyPolicy(TEDPolicy):


    # def get_default_config() -> Dict[Text, Any]:
    #     """Returns the component's default config.

    #     Default config and user config are merged by the `GraphNode` before the
    #     config is passed to the `create` and `load` method of the component.

    #     Returns:
    #         The default config of the component.
    #     """
    #     return {'priority': 6}

    # # @classmethod
    # # def _metadata_filename(cls):
    # #     return "my_policy_files.json"

    # # def _metadata(self):
    # #     return {}

    # def train(
    #     self,
    #     training_trackers: List[TrackerWithCachedStates],
    #     domain,
    #     **kwargs: Any,
    # ) -> Resource:
    #     return self._resource

    # def predict_action_probabilities(
    #     self, tracker: DialogueStateTracker, domain, rule_only_data: Optional[Dict[Text, Any]] = None, **kwargs: Any
    # ) -> List[float]:
    #     print("Hello")
    #     probs = [0.0] * domain.num_actions
    #     probs[0] = 1.0
    #     return self._prediction(probs)

    # def predict_action_probabilities(
    #     self,
    #     tracker: DialogueStateTracker,
    #     domain: Domain,
    #     rule_only_data: Optional[Dict[Text, Any]] = None,
    #     precomputations: Optional[MessageContainerForCoreFeaturization] = None,
    #     **kwargs: Any,
    # ) -> PolicyPrediction:
    #     print("I'm custom")
    #     super().predict_action_probabilities(tracker, domain, rule_only_data, precomputations, **kwargs)


    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        precomputations: Optional[MessageContainerForCoreFeaturization] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        probs = [0.0] * domain.num_actions
        probs[-1] = 1.0
        return PolicyPrediction(probs, policy_name="test_policy", policy_priority=6)