# Include necessary packages (torch, spacy, ...)
from selfcheckgpt.modeling_selfcheck import SelfCheckMQAG, SelfCheckBERTScore, SelfCheckNgram
import spacy
nlp = spacy.load('en_core_web_sm')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# selfcheck_mqag = SelfCheckMQAG(device=device) # set device to 'cuda' if GPU is available
selfcheck_bertscore = SelfCheckBERTScore(rescale_with_baseline=True)
# selfcheck_ngram = SelfCheckNgram(n=1) # n=1 means Unigram, n=2 means Bigram, etc.

# LLM's text (e.g. GPT-3 response) to be evaluated at the sentence level  & Split it into sentences
passage = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He is the host of The Savage Nation."
sentences = [sent.text.strip() for sent in nlp(passage).sents] # spacy sentence tokenization
print(sentences)
['Michael Alan Weiner (born March 31, 1942) is an American radio host.', 'He is the host of The Savage Nation.']

# Other samples generated by the same LLM to perform self-check for consistency
sample1 = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He is the host of The Savage Country."
sample2 = "Michael Alan Weiner (born January 13, 1960) is a Canadian radio host. He works at The New York Times."
sample3 = "Michael Alan Weiner (born March 31, 1942) is an American radio host. He obtained his PhD from MIT."

# --------------------------------------------------------------------------------------------------------------- #
# SelfCheck-MQAG: Score for each sentence where value is in [0.0, 1.0] and high value means non-factual
# Additional params for each scoring_method:
# -> counting: AT (answerability threshold, i.e. questions with answerability_score < AT are rejected)
# -> bayes: AT, beta1, beta2
# -> bayes_with_alpha: beta1, beta2


# sent_scores_mqag = selfcheck_mqag.predict(
#     sentences = sentences,               # list of sentences
#     passage = passage,                   # passage (before sentence-split)
#     sampled_passages = [sample1, sample2, sample3], # list of sampled passages
#     num_questions_per_sent = 5,          # number of questions to be drawn  
#     scoring_method = 'bayes_with_alpha', # options = 'counting', 'bayes', 'bayes_with_alpha'
#     beta1 = 0.8, beta2 = 0.8,            # additional params depending on scoring_method
# )
# print(sent_scores_mqag)
# [0.30990949 0.42376232]

# --------------------------------------------------------------------------------------------------------------- #
# SelfCheck-BERTScore: Score for each sentence where value is in [0.0, 1.0] and high value means non-factual
import time


sent_scores_bertscore = selfcheck_bertscore.predict(
    sentences = sentences,                          # list of sentences
    sampled_passages = [sample1, sample2, sample3], # list of sampled passages
)
print(sent_scores_bertscore)
# [0.0695562  0.45590915]

# --------------------------------------------------------------------------------------------------------------- #
# SelfCheck-Ngram: Score at sentence- and document-level where value is in [0.0, +inf) and high value means non-factual
# as opposed to SelfCheck-MQAG and SelfCheck-BERTScore, SelfCheck-Ngram's score is not bounded
# sent_scores_ngram = selfcheck_ngram.predict(
#     sentences = sentences,   
#     passage = passage,
#     sampled_passages = [sample1, sample2, sample3],
# )
# print(sent_scores_ngram)


# {'sent_level': { # sentence-level score similar to MQAG and BERTScore variant
#     'avg_neg_logprob': [3.184312, 3.279774],
#     'max_neg_logprob': [3.476098, 4.574710]
#     },
#  'doc_level': {  # document-level score such that avg_neg_logprob is computed over all tokens
#     'avg_neg_logprob': 3.218678904916201,
#     'avg_max_neg_logprob': 4.025404834169327
#     }
# }