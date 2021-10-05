import pdb
import sys
import os
import re
import time
import SentWrapper
MASK_TAG = "__entity__"
MASK_TAG2 = "[MASK]"

DISPATCH_MASK_TAG = "entity"
MODEL_PATH ='bert-large-cased'


#def read_common_descs(file_name):
#    common_descs = {}
#    with open(file_name) as fp:
#        for line in fp:
#            common_descs[line.strip()] = 1
##    print("Common descs for filtering read:",len(common_descs))
#    return common_descs
map_labels_file = 'data/map_labels.txt'


def read_ent_file(ent_file):
    file1 = open(ent_file, 'r')
    Lines = file1.readlines()
    word_ent_dict = {}
    # Strips the newline character
    for line in Lines:
        words = re.split('\s+', line)
        cur_ent = {}
        i = 1
        while i < len(words)-1:
            cur_ent[words[i]]=float(words[i+1])
            i += 2
        word_ent_dict[words[0]] = cur_ent
    ent_dict_adv = {}
    for key in word_ent_dict.keys():
        ent_dict_adv[key] = word_ent_dict[key]
        if key.lower() not in word_ent_dict.keys():
            ent_dict_adv[key.lower()] = word_ent_dict[key]
#        pdb.set_trace()
    
    return ent_dict_adv

class UnsupNER:
    def __init__(self,neigh=200, context_neigh=200, weight=5):
        self.desc_singleton = SentWrapper.SentWrapper()
#        self.desc_singleton.descs = SentWrapper.read_descs(DESC_FILE_ADV)
#        self.cluster_singleton = dist_v2.BertEmbeds('cache/',0,'data/vocab.txt','data/bert_vectors.txt',True,True,'data/labels.txt','data/stats_dict.txt','data/preserve_1_2_grams.txt','data/glue_words.txt')
#        self.common_descs = read_common_descs(cf.read_config()["COMMON_DESCS_FILE"])
        self.Word_Entities = read_ent_file('data/word_ent'+str(neigh)+'w'+str(weight)+'.txt')   # word_ent100_pure.txt
        self.Word_Entities_C = self.Word_Entities #dist_v2.read_ent_file('data/word_ent'+str(context_neigh)+'w'+str(weight)+'.txt')   # word_ent100_pure.txt

    def context_cue_new(self, masked_sent, bool_tweet_bert=0, ent=[],  ori_masked_sen = ''):
        ent_prob = {}
        ent_prob['LOC'] = 0
        ent_prob_gen = {}
        ent_prob_gen['LOC'] = 0
        descs = []
        if bool_tweet_bert:
#            start_time = time.time()
            descs, desc_probs = self.desc_singleton.punct_sentence_tweet(masked_sent)
#            print(bool_tweet_bert, 'punct_sentence_tweet: ', time.time()-start_time)

            if len(ent)  == 1:
                new_descs = []
                new_desc_probs = []
                for i, item in enumerate(descs):
                    if item.lower() !=  ent[0].lower():
                        new_descs.append(item)
                        new_desc_probs.append(desc_probs[i])
                descs = new_descs
                desc_probs = new_desc_probs
        else:
            if 1 < len(ent):
                masked_sent = masked_sent.replace(DISPATCH_MASK_TAG,MASK_TAG2)
#                start_time = time.time()

                descs, desc_probs = self.desc_singleton.punct_sentence_simple(masked_sent)
#                print(bool_tweet_bert, 'punct_sentence_simple2: ', time.time()-start_time)
                
            else:
#                start_time = time.time()
                if ent[0] in self.Word_Entities.keys():
                    return_dic = self.Word_Entities[ent[0]]
                    if 'LOC' not in return_dic.keys():
                        return_dic['LOC'] = 0
#                    print(bool_tweet_bert, 'bert cluster: ', time.time()-start_time)
                    return return_dic, return_dic,descs
                elif ent[0].lower() in self.Word_Entities.keys():
                    return_dic = self.Word_Entities[ent[0].lower()]
                    if 'LOC' not in return_dic.keys():
                        return_dic['LOC'] = 0
#                    print(bool_tweet_bert, 'bert cluster: ', time.time()-start_time)

                    return return_dic, return_dic,descs

                else:
                    
                    masked_sent = ori_masked_sen.replace(DISPATCH_MASK_TAG,MASK_TAG2)
#                    start_time = time.time()
                    descs, desc_probs = self.desc_singleton.punct_sentence_simple(masked_sent)
#                    print(bool_tweet_bert, 'punct_sentence_simple2: ', time.time()-start_time)

        print(descs)
      
        sum_pro = 0
        sum_pro_gen = 0
#        sum_entity = []
#        start_time = time.time()
#        pdb.set_trace()
        for j, desc in enumerate(descs):
#            print(desc)
            if bool_tweet_bert and 3 < len(str(desc)):
                desc = desc.capitalize()
            if desc not in self.desc_singleton.descs:
                sum_pro+=desc_probs[j]
#                sum_entity.append("0")
                sum_pro_gen += 1
            else:
                if desc in self.Word_Entities_C.keys() or desc.lower() in self.Word_Entities_C.keys():
                    if desc in self.Word_Entities_C.keys():
                        entities_raw = self.Word_Entities_C[desc]
                    else:
                        entities_raw = self.Word_Entities_C[desc.lower()]
                        
                    if bool_tweet_bert: # 
                        entities = entities_raw
                    else:
                        entities = {}
                        max_ent = max(entities_raw, key=entities_raw.get)
                        if  entities_raw[max_ent] < 0.1:
                            entities['LOC'] = 0
                        else:                            
                            entities[max_ent] = 1
                    for m in entities.keys():
                        if m:
                            if m in ent_prob.keys():
                                ent_prob[m]+= entities[m]*desc_probs[j]
                            else:
                                ent_prob[m] = entities[m]*desc_probs[j]
                                
                            if m in ent_prob_gen.keys():
                                ent_prob_gen[m] += entities[m]
                            else:
                                ent_prob_gen[m] = entities[m]
                                
                        sum_pro+=entities[m]*desc_probs[j]
                        sum_pro_gen += ent_prob_gen[m]
    
                else:
                    sum_pro+=desc_probs[j]
                    sum_pro_gen += 1
    
        if sum_pro:
            for key in ent_prob.keys():
                ent_prob[key]=ent_prob[key]/sum_pro

        if sum_pro_gen:
            for key in ent_prob_gen.keys():
                ent_prob_gen[key]=ent_prob_gen[key]/sum_pro_gen
                
        return ent_prob, ent_prob_gen,descs





