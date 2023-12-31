# import evaluate
#
# bert_score = evaluate.load("/public/home/zhangke/rgrg-main/src/full_model/evaluate_full_model/bert_score/")

import evaluate_full_model.bert_score.bert_score as bert_score
cands = ['afe']
refs = ['kmf']
bert_score_results = bert_score.score(cands, refs, lang="en", model_type="distilbert-base-uncased")
print(bert_score_results, type(bert_score_results))
