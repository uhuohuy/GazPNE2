#from pytorch_transformers import *
import numpy as np
from string import ascii_letters, digits
char_list = [c for c in ascii_letters]
import re
import logging
logging.basicConfig(level=logging.INFO)

import os
import sys

from transformers import pipeline
#print('import pipeline', time.time()-st)
#st = time.time()

from fairseq.models.roberta import RobertaModel
#print('import RobertaModel', time.time()-st)
#st = time.time()

# Incorporate the BPE encoder into BERTweet-base 
from fairseq.data.encoders.fastbpe import fastBPE  
#print('import fastBPE', time.time()-st)
#st = time.time()

from fairseq import options  
#print('import options', time.time()-st)
#st = time.time()

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))
#print('get_script_path', time.time()-st)

#pdb.set_trace()
			
		
top_k = 300
return_top_k = 40
DESC_FILE="data/common_desc2.txt"
TWEET_REMOVE_FILE="data/tweet_desc_remove.txt"

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')

def read_descs(file_name):
    ret_dict = {}
    with open(file_name) as fp:
        line = fp.readline().rstrip("\n")
        if (len(line) >= 1):
            ret_dict[line] = 1
        while line:
            line = fp.readline().rstrip("\n")
            if (len(line) >= 1):
                ret_dict[line] = 1
    return ret_dict

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class SentWrapper:
    def __init__(self):
#        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base",do_lower_case=False)
#        self.model = AutoModel.from_pretrained("vinai/bertweet-base")
#        st = time.time()
#
#
#        self.tokenizer = BertTokenizer.from_pretrained(path,do_lower_case=False) ### Set this to to True for uncased models
##        print('BertTokenizer.from_pretrained', time.time()-st)
##        st = time.time()
##
#        self.model = BertForMaskedLM.from_pretrained(path)
##        print('BertForMaskedLM.from_pretrained', time.time()-st)
##        st = time.time()
#
#        self.model.eval()
#        print('model.eval()', time.time()-st)
#        st = time.time()

        self.descs = read_descs(DESC_FILE)
#        print('read_descs', time.time()-st)
#        st = time.time()

        self.tweet_rm_descs = read_descs(TWEET_REMOVE_FILE)
#        print('tweet_rm_descs', time.time()-st)
#        st = time.time()

        self.BERTweet = RobertaModel.from_pretrained('BERTweet_base_fairseq', checkpoint_file='model.pt')
#        print('self.BERTweet', time.time()-st)
#        st = time.time()

        self.BERTweet.eval()  # disable dropout (or leave in train mode to finetune)
#        print('.eval()', time.time()-st)
#        st = time.time()

#        print('eval')
#        pdb.set_trace()
        self.unmasker = pipeline('fill-mask', model='bert-large-cased-whole-word-masking',topk=top_k)
        parser = options.get_preprocessing_parser()

        parser.add_argument('--bpe-codes', type=str, help='path to fastBPE BPE', default="BERTweet_base_fairseq/bpe.codes") 
        args, unknown = parser.parse_known_args()
        self.BERTweet.bpe = fastBPE(args) #Incorporate the BPE encoder into BERTweet
#
#        self.bert_model = AutoModelWithLMHead.from_pretrained("bert-large-cased-whole-word-masking")
#        print('self.bert_model', time.time()-st)
#        st = time.time()

        #pdb.set_trace()
    def punct_sentence_simple(self,text):

        text+='.'
        result = []
        result_probs = []
        debug_count = 0
        topk_filled_outputs = self.unmasker(text)
#        print('punct_sentence_simple unmasker: ', time.time()-start_time)
#        start_time = time.time()

        for candidate in topk_filled_outputs:
            new_token = candidate['token_str']
            if set(new_token).difference(ascii_letters + digits) or (new_token.lower() \
                   in self.tweet_rm_descs) or new_token.isdecimal() or new_token in char_list \
                   or (4 > len(new_token)): #or hasNumbers(new_token)
#            if (candidate[2] not in self.descs):
                continue
            if (debug_count < return_top_k):
#                 if new_token.islower():
#                     new_token = new_token.capitalize()
                 result.append(new_token)
                 result_probs.append( candidate['score'] )
                 debug_count += 1
#            k += 1
#            if (k >= top_k):
#                break
##                    print(result)
#        print('punct_sentence_simple candidate: ', time.time()-start_time)
                 
        if not result:
            return [0], [1]
        else:
            return result, softmax(result_probs)#


    
    def punct_sentence_tweet(self,text):
        result = []
        result_probs = []
        debug_count = 0
        topk_filled_outputs = self.BERTweet.fill_mask(text, topk=top_k)  
        for candidate in topk_filled_outputs:
            new_token = candidate[2].replace('#', '')
            if set(new_token).difference(ascii_letters + digits) or \
            (new_token.lower() in self.tweet_rm_descs) or new_token.isdecimal() \
            or new_token in char_list or  (4 > len(new_token)): #or hasNumbers(new_token) 
#            if (candidate[2] not in self.descs):
                continue
            if (debug_count < return_top_k):
#                 if new_token.islower():
#                     new_token = new_token.capitalize()
                 result.append(new_token)
                 result_probs.append(candidate[1])
                 debug_count += 1
#            k += 1
#            if (k >= top_k):
#                break
##                    print(result)
        if len(result) < 10:
            return ['0'], [1]
        else:
            return result, softmax(result_probs)#


#

def main():
    MODEL_PATH='bert-large-cased'
    singleton = SentWrapper(MODEL_PATH)
    out = singleton.punct_sentence("Apocalypse is a entity")
    print(out)


if __name__ == '__main__':
    # main()
    print(softmax([0.91,0.03,0.05,0.01]))
    
    print(set('12').difference(ascii_letters + digits))