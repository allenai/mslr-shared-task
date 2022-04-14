from evaluator import read_targets, read_predictions, calculate_rouge
import unittest


class TestEvaluator(unittest.TestCase):

    def test_eval_rouge(self):
        """
        Test that ROUGE works
        :return:
        """
        target_file = 'test-summaries.jsonl'
        generated_file = 'test-generated.jsonl'

        targets = read_targets(target_file)
        predictions = read_predictions(generated_file)
        rouge = calculate_rouge(targets, predictions)

        assert rouge
        assert rouge.get('rouge1')
        assert rouge.get('rouge2')
        assert rouge.get('rougeL')
        assert rouge.get('rougeLsum')

        assert rouge.get('rouge1').low.fmeasure < 0.26
        assert rouge.get('rouge1').high.fmeasure > 0.95

        assert rouge.get('rougeL').low.fmeasure < 0.26
        assert rouge.get('rougeL').high.fmeasure > 0.95

    # TODO: test Evidence inference