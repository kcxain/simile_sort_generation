from evaluation.evaluator import Evaluator
from model.bert_metric import BERTMetric

evale = Evaluator('output/model_best_dual_mlr_loss.ckpt')
model = BERTMetric()
print(evale.evaluate(model))