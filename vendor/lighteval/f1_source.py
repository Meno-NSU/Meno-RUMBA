# Extract from LightEval v0.13.0, MIT license; provenance in README.md.
# https://github.com/huggingface/lighteval/blob/v0.13.0/src/lighteval/metrics/metrics_sample.py
class F1_score(SampleLevelComputation):
    def __init__(
        self,
        aggregation_function: Callable[[list[float]], float] = max,
        normalize_gold: Callable[[str], str] | None = None,
        normalize_pred: Callable[[str], str] | None = None,
        strip_strings: bool = False,
    ):
        """An F1 score class. F1 is computed over the bag of words of the golds and predictions.

        Args:
            aggregation_function (callable, optional): How to aggregate the item results. Defaults to max.
                Used if there are several golds or predictions on which scores were computed.
            normalize_gold (callable, optional): Function to use to normalize the reference strings.
                Defaults to None if no normalization is applied.
            normalize_pred (callable, optional): Function to use to normalize the predicted strings.
                Defaults to None if no normalization is applied.
            strip_strings (bool, optional): Whether to strip both reference and predictions. Defaults to False.
        """
        if aggregation_function is None:
            aggregation_function = max

        self.aggregation_function = aggregation_function
        self.normalize_gold = normalize_gold
        self.normalize_pred = normalize_pred
        self.strip_strings = strip_strings

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        """Computes the metric over a list of golds and predictions for one single sample.

        Args:
            doc (Doc): The document containing gold references.
            model_response (ModelResponse): The model's response containing predictions.
            **kwargs: Additional keyword arguments.

        Returns:
            float: Aggregated score over the current sample's items.
        """
        results = []
        golds = doc.get_golds()
        predictions = model_response.final_text
        # We might need to flatten golds if they are a list of lists
        for gold in golds:
            for pred in predictions:
                results.append(self.compute_one_item(gold=gold, pred=pred))
        return self.aggregation_function(results)

    def compute_one_item(self, gold: str, pred: str) -> float:
        """Compares two strings only.

        Args:
            gold (str): One of the possible references
            pred (str): One of the possible predictions

        Returns:
            float: The f1 score over the bag of words, computed using nltk.
        """
        if self.normalize_gold:
            gold = self.normalize_gold(gold)

        if self.normalize_pred:
            pred = self.normalize_pred(pred)

        gold_bow = set(gold.split())
        pred_bow = set(pred.split())

        ret = nltk.scores.f_measure(gold_bow, pred_bow)

        if ret is None:
            return 0.0
        return ret


