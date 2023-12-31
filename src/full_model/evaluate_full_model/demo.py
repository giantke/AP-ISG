# import bert_score_source.bert_score_source.score as bert_score_source

import sys
import os

print(os.path.abspath(__file__))
print(os.path.dirname(os.path.abspath(__file__)))
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from bert_score_source.bert_score import score as bert_score

# cands = [['afe'],['adfe']]
# refs = [['kmf'],['adfe']]
cands = ['afe']
refs = ['kmf']
bert_score_results = bert_score(cands, refs, lang="en", model_type="distilbert-base-uncased")
print(bert_score_results, type(bert_score_results))
