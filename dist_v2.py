#import pdb
import sys
#import operator
from collections import OrderedDict
#import subprocess
import numpy as  np
import json
import math
#from pytorch_transformers import *
#import sys
from string import ascii_letters, digits
import os
import re

SINGLETONS_TAG  = "_singletons_ "
EMPTY_TAG = "_empty_ "
OTHER_TAG = "OTHER"
AMBIGUOUS = "AMB"
TOP_ENTITY_COUNT = 500
BERT_TERMS_START=106
UNK_ID = 100
#Original setting for cluster generation. 
OLD_FORM=True

try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')
    
#importent_ner = ['LOC','PER','ORG','TIME','SPORT','ENT','RELIGION','DISEASE','THING','DISCIPLINE','UNIV', 'LANGUAGE','SEQUENCE','PRODUCT','PROTEIN','GENE','COMPANY','GOV','UNITS','DRUG','NUMBER']

map_labels_file = 'data/map_labels.txt'

def read_embeddings(embeds_file):
    with open(embeds_file) as fp:
        embeds_dict = json.loads(fp.read())
    return embeds_dict

def count_list_frequency_proportion(list_data,sim,weight):
    stat_frequency = {}
    stat_proportion = {}
    if weight==0:
        weights = [1]*len(list_data)

    elif weight==1:
        weights=[d/sum(sim) for d in sim ]
    elif weight == 2:
        weights = softmax(sim)
    else:
        weights = inves_max(sim, weight-1)
    total = sum(weights)
    for i, e in enumerate(list_data):
        if str(e) in stat_frequency:
            stat_frequency[str(e)] += weights[i]
        else:
            stat_frequency[str(e)] = weights[i]
    for key, value in stat_frequency.items():
        stat_proportion[key] = value / total
        
    return stat_frequency, stat_proportion

def read_labels(labels_file):
    terms_dict = OrderedDict()
    with open(labels_file) as fin:
        count = 1
        for term in fin:
            term = term.strip("\n")
            term = term.split()
            if (len(term) == 5):
                terms_dict[term[2]] = {"label":term[0],"aux_label":term[1],"mean":float(term[3]),"variance":float(term[4])}
                count += 1
            else:
                print("Invalid line:",term)
                assert(0)
#    print("count of labels in " + labels_file + ":", len(terms_dict))
    return terms_dict

def read_all_labels(map_labels):
    terms_dict = dict()
    with open(map_labels) as maplab:
#        count = 1
#        maplab_values=[]
#        debug_values=[]

        for term1 in maplab:
            term1=term1.replace("'","")
            term1=term1.replace("[","")
            term1=term1.replace("]","")
            term1=term1.replace(",","")
            term1 = term1.strip("\n")
            term1 = term1.split()
            terms_dict[term1[2]] = {"label":term1[0],"aux_label":term1[1]} #,"mean":float(term[5]),"variance":float(term[6])
            for i in range(7,len(term1)):
                terms_dict[term1[i]] = {"label":term1[0],"aux_label":term1[1]} #,"mean":float(term[5]),"variance":float(term[6])
    return terms_dict


        
def read_terms(terms_file):
    terms_dict = OrderedDict()
    with open(terms_file) as fin:
        count = 1
        for term in fin:
            term = re.split('\s+', term)
#            term = term.strip("\n")
#            if (len(term) > 1):
#                terms_dict[term[0]] = int(term[1])
#            else:
            terms_dict[term[0]] = count
            count += 1

#    print("count of tokens in ",terms_file,":", len(terms_dict))
    return terms_dict

def is_filtered_term(key): #Words selector. skiping all unused and special tokens
    if (OLD_FORM): 
        return True if (str(key).startswith('#') or str(key).startswith('[')) else False
    else:
        return True if (str(key).startswith('[')) else False

def filter_2g(term,preserve_dict):
    if (OLD_FORM):
        return True if  (len(term) <= 2 ) else False
    else:
        return True if  (len(term) <= 2 and term not in preserve_dict) else False

class BertEmbeds:
    def __init__(self, model_path,do_lower, terms_file,embeds_file,cache_embeds,normalize,labels_file,stats_file,preserve_2g_file,glue_words_file):
        do_lower = True if do_lower == 1 else False
        self.tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=do_lower)
        self.terms_dict = read_terms(terms_file)
        
        self.labels_dict = read_labels(labels_file)
        self.entire_labels = read_all_labels(map_labels_file)
#        pdb.set_trace()
        self.stats_dict = read_terms(stats_file)
        self.preserve_dict = read_terms(preserve_2g_file)
        self.gw_dict = read_terms(glue_words_file)
        self.embeddings = read_embeddings(embeds_file)
        self.cache = cache_embeds
        self.embeds_cache = {}
        self.cosine_cache = {}
        self.dist_threshold_cache = {}
        self.normalize = normalize


    def gen_pivot_graphs(self,threshold,count_limit):
        tokenize = False
        count = 1
        picked_dict = OrderedDict()
        pivots_dict = OrderedDict()
        singletons_arr = []
        empty_arr = []
        total = len(self.terms_dict)
        dfp = open(str(threshold)+"debug_pivots.txt","w")
        for key in self.terms_dict:
            if (is_filtered_term(key) or count <= BERT_TERMS_START):
                count += 1
                continue
            count += 1
            #print(":",key)
            if (key in picked_dict or len(key) <= 2):
                continue
#            print("Processing ",count," of ",total)
            picked_dict[key] = 1
            sorted_d = self.get_terms_above_threshold(key,threshold,tokenize)
            arr = []
            for k in sorted_d:
                if (is_filtered_term(k) or filter_2g(k,self.preserve_dict)):
                    picked_dict[k] = 1
                    continue
                if (sorted_d[k] < count_limit):
                    picked_dict[k] = 1
                    arr.append(k)
                else:
                    break
            if (len(arr) > 1):
                max_mean_term,max_mean, std_dev,s_dict = self.find_pivot_subgraph(arr,tokenize)
                if (max_mean_term not in pivots_dict):
                    new_key  = max_mean_term
                else:
                    print("****Term already a pivot node:",max_mean_term, "key  is :",key)
                    new_key  = max_mean_term + "++" + key
                pivots_dict[new_key] = {"key":new_key,"orig":key,"mean":max_mean,"terms":arr}
                print(new_key,max_mean,std_dev,arr)
                dfp.write(new_key + " " + new_key + " " + new_key+" "+key+" "+str(max_mean)+" "+ str(std_dev) + " " +str(arr)+"\n")
            else:
                if (len(arr) == 1):
                    print("***Singleton arr for term:",key)
                    singletons_arr.append(key)
                else:
                    print("***Empty arr for term:",key)
                    empty_arr.append(key)

        dfp.write(SINGLETONS_TAG + str(singletons_arr) + "\n")
        dfp.write(EMPTY_TAG + str(empty_arr) + "\n")
        with open("pivots.json","w") as fp:
            fp.write(json.dumps(pivots_dict))
        dfp.close()


    def gen_dist_for_vocabs(self):
        count = 1
        picked_count = 0
        cum_dict = OrderedDict()
        cum_dict_count = OrderedDict()
        for key in self.terms_dict:
            if (is_filtered_term(key) or count <= BERT_TERMS_START):
                count += 1
                continue
            #print(":",key)
            picked_count += 1
            sorted_d = self.get_distribution_for_term(key,False)
            for k in sorted_d:
                val = round(float(k),1)
                #print(str(val)+","+str(sorted_d[k]))
                if (val in cum_dict):
                    cum_dict[val] += sorted_d[k]
                    cum_dict_count[val] += 1
                else:
                    cum_dict[val] = sorted_d[k]
                    cum_dict_count[val] = 1
        for k in cum_dict:
            cum_dict[k] = float(cum_dict[k])/cum_dict_count[k]
        final_sorted_d = OrderedDict(sorted(cum_dict.items(), key=lambda kv: kv[0], reverse=False))
        print("Total picked:",picked_count)
        with open("cum_dist.txt","w") as fp:
            fp.write("Total picked:" + str(picked_count) + "\n")
            for k in final_sorted_d:
                print(k,final_sorted_d[k])
                p_str = str(k) + " " +  str(final_sorted_d[k]) + "\n"
                fp.write(p_str)

    def get_embedding(self,text,tokenize=True):
        if (self.cache and text in self.embeds_cache):
            return self.embeds_cache[text]
        if (tokenize):
            tokenized_text = self.tokenizer.tokenize(text)
        else:
            tokenized_text = text.split()
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        vec =  self.get_vector(indexed_tokens)
        if (self.cache):
                self.embeds_cache[text] = vec
        return vec


    def get_vector(self,indexed_tokens):
        vec = None
        if (len(indexed_tokens) == 0):
            return vec
        #pdb.set_trace()
        for i in range(len(indexed_tokens)):
            term_vec = self.embeddings[indexed_tokens[i]]
            if (vec is None):
                vec = np.zeros(len(term_vec))
            vec += term_vec
        sq_sum = 0
        for i in range(len(vec)):
            sq_sum += vec[i]*vec[i]
        sq_sum = math.sqrt(sq_sum)
        for i in range(len(vec)):
            vec[i] = vec[i]/sq_sum
        return vec

    def calc_inner_prod(self,text1,text2,tokenize):
        if (self.cache and text1 in self.cosine_cache and text2 in self.cosine_cache[text1]):
            return self.cosine_cache[text1][text2]
        vec1 = self.get_embedding(text1,tokenize)
        vec2 = self.get_embedding(text2,tokenize)
        if (vec1 is None or vec2 is None):
            return 0
        val = np.inner(vec1,vec2)
        if (self.cache):
            if (text1 not in self.cosine_cache):
                self.cosine_cache[text1] = {}
            self.cosine_cache[text1][text2] = val
        return val

    def get_distribution_for_term(self,term1,tokenize):
        debug_fp = None
        hack_check = False

        if (term1 in self.dist_threshold_cache):
            return self.dist_threshold_cache[term1]
        dist_dict = {}
        if (hack_check and debug_fp is None):
            debug_fp = open("debug.txt","w")
        for k in self.terms_dict:
            term2 = k.strip("\n")
            val = self.calc_inner_prod(term1,term2,tokenize)
            if (hack_check and val >= .6 and val < .8 and term1 != term2):
                str_val = term1 + " " + term2 + "\n"
                debug_fp.write(str_val)
                debug_fp.flush()

            val = round(val,2)
            if (val in dist_dict):
                dist_dict[val] += 1
            else:
                dist_dict[val] = 1
        sorted_d = OrderedDict(sorted(dist_dict.items(), key=lambda kv: kv[0], reverse=False))
        self.dist_threshold_cache[term1] = sorted_d
        return sorted_d

    def get_terms_above_threshold(self,term1,threshold,tokenize):
        final_dict = {}
        for k in self.terms_dict:
            term2 = k.strip("\n")
            val = self.calc_inner_prod(term1,term2,tokenize)
            val = round(val,2)
            if (val > threshold):
                final_dict[term2] = val
        sorted_d = OrderedDict(sorted(final_dict.items(), key=lambda kv: kv[1], reverse=True))
        return sorted_d


    #given n terms, find the mean of the connection strengths of subgraphs considering each term as pivot.
    #return the mean of max strength term subgraph
    def find_pivot_subgraph(self,terms,tokenize):
        max_mean = -1000
        std_dev = 0
        max_mean_term = None
        means_dict = {}
        if (len(terms) == 1):
            return terms[0],1,0,{terms[0]:1}
#        print('terms ', terms)
        for i in terms:
            full_score = 0
            count = 0
            full_dict = {}
            for j in terms:
                if (i != j):
#                    print('i, j ', i, j)
#                    if (i=='I' and j=='Sure'):
#                        pdb.set_trace()
                    val = self.calc_inner_prod(i,j,tokenize)
                    #print(i+"-"+j,val)
                    full_score += val
                    full_dict[count] = val
                    count += 1
            if (len(full_dict) > 0):
                mean  =  float(full_score)/len(full_dict)
                means_dict[i] = mean
                #print(i,mean)
                if (mean > max_mean):
                    #print("MAX MEAN:",i)
                    max_mean_term = i
                    max_mean = mean
                    std_dev = 0
                    for k in full_dict:
                        std_dev +=  (full_dict[k] - mean)*(full_dict[k] - mean)
                    std_dev = math.sqrt(std_dev/len(full_dict))
                    #print("MEAN:",i,mean,std_dev)
        #print("MAX MEAN TERM:",max_mean_term)
        sorted_d = OrderedDict(sorted(means_dict.items(), key=lambda kv: kv[1], reverse=True))
        return max_mean_term,round(max_mean,2),round(std_dev,2),sorted_d

    
    def gen_label(self,node):
        if (node["label"] in self.stats_dict):
            if (node["aux_label"] in self.stats_dict):
                ret_label = node["label"] + "-" +  node["aux_label"]
            else:
                if (node["label"]  == AMBIGUOUS):
                    ret_label = node["label"] + "-" +  node["aux_label"]
                else:
                    ret_label = node["label"]
        else:
                ret_label = OTHER_TAG + "-" + node["label"]
        return ret_label

    def filter_glue_words(self,words):
        ret_words = []
        for dummy,i in enumerate(words):
            if (i not in self.gw_dict):
                ret_words.append(i)
        if (len(ret_words) == 0):
            ret_words.append(words[0])
        return ret_words
    
#    def filter_desc(self, desc):
#        new_desc = {}
#        for key in desc.keys():
#            entities = self.find_entities([key])
#            if entities[0] in importent_ner:
#                new_desc[key]=desc[key]
#        return new_desc

    def find_entities(self,words):
        entities = self.labels_dict
        tokenize = False
        words = self.filter_glue_words(words)
        desc_max_term,desc_mean,desc_std_dev,s_dict = self.find_pivot_subgraph(words,tokenize)
        pivot_similarities = {}
        for i,key in enumerate(entities):
            term = key
            val = round(self.calc_inner_prod(desc_max_term,term,tokenize),2)
            pivot_similarities[key] = val
        sorted_d = OrderedDict(sorted(pivot_similarities.items(), key=lambda kv: kv[1], reverse=True))
        count = 0
        ret_arr = []
        for k in sorted_d:
            ret_label = self.gen_label(entities[k])

            ret_arr.append(ret_label)
            count+= 1
            if (count >= 1):
                break
        return ret_arr
    
    def pre_cal_entities(self,terms_file1,terms_file2,saved_file, ent_num=1,weight=0):
        saved_file += str(ent_num) +'w'+str(weight)+'.txt'
        if os.path.isfile(saved_file):
            os.remove(saved_file)
        print(terms_file1,terms_file2,saved_file)
        bert_terms_dict = read_terms(terms_file1)
        total_words = list(bert_terms_dict.keys())
#        tweet_terms_dict = read_terms(terms_file2)
#        total_words.extend(list(tweet_terms_dict.keys()))

        total_words = set(total_words)
        word_count = 0
        entities = self.labels_dict
#        print('total word count:', len(total_words))
        with open(saved_file, 'a') as the_file:
            for word in total_words:
                if is_filtered_term(word) or set(word).difference(ascii_letters + digits):
                    print(word)
                    continue
                word_count += 1
                print(word_count)
                words = [word]
                tokenize = False
                words = self.filter_glue_words(words)
                desc_max_term,desc_mean,desc_std_dev,s_dict = self.find_pivot_subgraph(words,tokenize)
                pivot_similarities = {}
                for i,key in enumerate(entities):
                    term = key
                    val = round(self.calc_inner_prod(desc_max_term,term,tokenize),2)
                    pivot_similarities[key] = val
                sorted_d = OrderedDict(sorted(pivot_similarities.items(), key=lambda kv: kv[1], reverse=True))
                count = 0
                ret_arr = []
                similarities = []
                close_words = []
                for k in sorted_d:
                    ret_label = self.gen_label(entities[k])
                    close_words.append(entities[k]["aux_label"])
                    ret_arr.append(ret_label)
                    similarities.append(pivot_similarities[k])
                    count+= 1
                    if (count >= ent_num):
                        break

                fre, prop = count_list_frequency_proportion(ret_arr,similarities,weight)
                sorted_d = OrderedDict(sorted(prop.items(), key=lambda kv: kv[1], reverse=True))
                write_str = word + ' '
                sum_count = 0
                for key in sorted_d:
                    if sum_count < TOP_ENTITY_COUNT:
                        sum_count += 1
                        write_str += key + ' ' + str(sorted_d[key]) + ' '
                    else:
                        break
                the_file.write( write_str +'\n')
        

    def find_entities_simple(self,words):
        entities = self.entire_labels
        words = self.filter_glue_words(words)
        ret_arr = []
        if words:
            if words[0] in entities.keys():
                ret_label = self.gen_label(entities[words[0]])
                ret_arr.append(ret_label)
        return ret_arr


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def inves_max(x, p=2.5):
    """Compute softmax values for each sets of scores in x."""
    temp = []
    for item in x:
        temp.append(pow(item,p))
    sum_temp = sum(temp)
    temp = [item /sum_temp for item in temp]
    return temp


def main():
    if (len(sys.argv) != 12 and len(sys.argv) != 16):
        print("Usage: <Bert model path - to load tokenizer> do_lower_case[1/0] <vocab file> <vector file> <tokenize text>1/0 <labels_file>  <preserve_1_2_grams_file> < glue words file>")
    else:
        tokenize = True if int(sys.argv[5]) == 1 else False
        if (tokenize == True):
            print("Forcing tokenize to false. Ignoring input value")
            tokenize = False #Adding this override to avoid inadvertant subword token generation error for pivot cluster generation
        print("Tokenize is set to :",tokenize)
        b_embeds =BertEmbeds(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],True,True,sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9]) #True - for cache embeds; normalize - True
        display_threshold = .4
        while (True):
            print("Enter test type (0-gen cum dist for vocabs (will take approx 3 hours); 1-generate clusters (will take approx 2 hours);  2-neigh/3-pivot graph/4-bipartite/5-Entity test: q to quit")
            val = sys.argv[10]
            if (val == "0"):
                b_embeds.gen_dist_for_vocabs()
                sys.exit(-1)
            elif (val == "1"):
                print("Enter Input threshold .5  works well for both pretraining and fine tuned")
                val = float(sys.argv[11])
                tail = 10
                print("Using value: ",val)
                b_embeds.gen_pivot_graphs(val,tail)
                sys.exit(-1)
            elif (val == 'q'):
                sys.exit(-1)
            elif (val == '6'):
                b_embeds.pre_cal_entities(sys.argv[3],sys.argv[12],sys.argv[13],int(sys.argv[14]),int(sys.argv[15]))
                sys.exit(-1)
            else:
                print("invalid option")


if __name__ == '__main__':
    main()
