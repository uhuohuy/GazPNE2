import re
import codecs
import unicodedata
import csv
import numpy as np
import re
import pdb
from collections import OrderedDict
PERSON_POS_ADV = ['^','NNP','NNPS']
long_places_pos = ['^','$','CD','NNP','NNPS']
#'spring', 'ottawa',
category_words = [ 'spur',  'hollow',  'farm', 'mdws',  'pond', 'hill', 'highway', 'hall', 'airport', 'canal', '\
vsta', 'aprt', 'pkwy', 'temple',  'beck', 'flat',  'lake',  'blvd', 'reservoir', \
'canyon', 'terrace', 'beach', 'crescent',  'academy', 'street',  'centre',  'range', 'bdge', 'lane',\
'ridge', 'lower',  'island', 'club', 'ranch',  'colony', 'court', 'west', 'upper', 'grange', \
'northeast',  'field',  'southwest',  'manor', 'crss', 'gte', 'south', 'boulevard', 'cove', 'cottage',  \
'croft', 'road', 'grove', 'city',  'hospital', 'link',  'trail', 'park', 'crest', 'state', 'chase', 'top', 'market', 'university',\
'northwest', 'brook', 'cres', 'dale', 'lwr', 'mountain', 'haven', 'bank', \
 'pike', 'vista', 'vill', 'meadow', 'shwy', 'comm', 'circle',  'north', 'college', 'school', 'parkway', \
 'station', 'center', 'centre' 'cott', 'lodge',  'branch', 'plza', 'river', 'knoll', 'lough', \
 'cross',   'wood', 'village',  'bluff',  'church', 'bridge', 'plaza', \
 'southeast', 'rdge',  'center', 'loop', 'east', 'town',  'view', 'county',  'brdg', 'city', 'rnge', 'forest', 'mill',\
 'point', 'woods',  'mesa',  'green', 'division', 'glen', 'rock', 'vale', 'township', 'crst', 'meadows',\
 'place', 'creek',  'water', 'univer', 'mount',  'hosp', 'terr', 'garden', 'univ', 'mall' 'orchard', \
 'avenue',  'valley', 'hotel','zoo', 'roads','streets','rivers','area', 'sea','city','town','roads', 'parks','countries','southern',\
 'northern','house','home','northeast','eastern','stream','streams','bayou', 'amp', 'plains', 'river','south','nw','nb','eb','sb','wb','westbound','northbound','eastbound','southbound']
prefix_places_words = ['west','north','east','south','northeast','southeast','northwest','southwest','central','northern','southern','eastern','western'\
                       'w','e','s','ne','se','nw','sw','northeastern','southeastern','northwestern','southwestern','eb','sb','wb','nb','eastbound','southbound','westbound','northbound']
lastfix_places_words = prefix_places_words; #['eb','sb','wb','nb','eastbound','southbound','westbound','northbound']
categories = category_words
categories.extend(prefix_places_words)
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

'''load the word embedding from a file'''
def load_embeding(emb_file):
    vectors = []
    words = []
    idx = 0
    word2idx = {}
    with open(emb_file, 'rb') as f:
        for l in f:
            line = l.decode().split()
            if len(line) > 5:
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)
                emb_dim = len(vect)
    glove = {w: vectors[word2idx[w]] for w in words}
    return glove,emb_dim

def load_embeding1(emb_file):
    # words = []
    idx = 0
    word2idx = {}
    glove = {}
    with open(emb_file, 'rb') as f:
        for l in f:
            line = l.split()
            if len(line) > 5:
                word = line[0]
                # words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = [float(line[i]) for i in range(1,len(line))]#np.array(line[1:]).astype(np.float)
                # vectors.append(vect)
                glove[word]=vect
    emb_dim = len(vect)
    
    # glove = {w: vectors[word2idx[w]] for w in words}
    return glove,emb_dim

def load_bigram_model(bigram_file):
    bigram_model = {}

    with open(bigram_file, 'rb') as f:
        for l in f:
            line = l.decode().split()
            if len(line) == 3:
                bigram_model[(line[0],line[1])] = float(line[2])
    return bigram_model

def load_bigram_model1(bigram_file):
    bigram_model = {}
    with open(bigram_file, 'rb') as f:
        for l in f:
            line = l.decode().split()
            bigram_model[(line[0],line[1])] = float(line[2])
    return bigram_model


def save_bigram_model(listOfBigrams,file_name):
    file = open(file_name, 'w')
    for bigrams in listOfBigrams:
        value = round(listOfBigrams[bigrams], 5)
        if value == 0.0:
            value = 0
        if value==1.0:
            value = 1
        file.write(bigrams[0]+ ' ' + bigrams[1] + ' ' + str(value) + '\n')
    file.close()

def feat_char_loc(word, max_char_len):
    return_char_loc_vector = []
    for i in range(max_char_len-len(word)):
        return_char_loc_vector.append(0)
        return_char_loc_vector.append(0)
    for i, c in enumerate(word):
        return_char_loc_vector.append(i+1)
        return_char_loc_vector.append(len(word)-i)
    return return_char_loc_vector

def using_split3(line, reg):
    words = re.split(reg,line)
#    words = line.split()
    index = line.index
    offsets = []
    append = offsets.append
    running_offset = 0
    for word in words:
        word_offset = index(word, running_offset)
        word_len = len(word)
        running_offset = word_offset + word_len
        append((word, word_offset, running_offset - 1))
    return offsets

''' return the intersetion set of two lists'''
def interset(list1,list2):
    return_list = []
    for l1 in list1:
        try:
            index = list2.index(l1)
            return_list.append(l1)
            list2.remove(index)
        except ValueError:
            continue
    return return_list




def sub_lists_adv(list1, list_index, max_len): 
    # store all the sublists  
    sublist = []
    sublist_index = []   
    # first loop  
    for i in range(len(list1) + 1):         
        # second loop  
        for j in range(i + 1, len(list1) + 1):             
            # slice the subarray
            if j-i<max_len:
                sub = list1[i:j]
                sub_index = list_index[i:j]
                sublist_index.append(sub_index)
                sublist.append(sub)
    return sublist,sublist_index



''' get sub list of a list with the sub list length below max_lem'''
def sub_lists(list1,max_len): 
    # store all the sublists  
    sublist = []
    sublist_index = []   
    # first loop  
    for i in range(len(list1) + 1):         
        # second loop  
        for j in range(i + 1, len(list1) + 1):             
            # slice the subarray
            if j-i<max_len:
                sub = list1[i:j]
                sub_index = range(i,j)
                sublist_index.append(list(sub_index))
                
                sublist.append(sub)
    return sublist,sublist_index

def index_list_int(index, index_list, sub_index):
    bool_intersect = 0
    for i in index_list:
        if intersection(sub_index[index],sub_index[i]):
            bool_intersect = 1
            break
    return bool_intersect

def index_list_sub(index, index_list, sub_index):
    return_sub = []
    for i in index_list:
        if is_Sublist(sub_index[index],sub_index[i]):
            return_sub.append(i)
        elif is_Sublist(sub_index[i],sub_index[index]):
            return_sub.append(index)
    return list(set(return_sub))

def offinoffset(cur_off_pla, hashtag_offsets):
    for off in hashtag_offsets:
        if cur_off_pla[0] >= off[0] and cur_off_pla[0] <= off[1] and \
           cur_off_pla[1] >= off[0] and cur_off_pla[1] <= off[1]:
               return True
    return False


def interset_adv(list1,list2):
    first_place = ''
    second_place = ''
    for i in list1:
        first_place += i.lower()
    for j in list2:
        second_place += j.lower()
    if first_place==second_place:
        match = 1
    else:
        match = 0
    return match

def str2TupleList(test_str, bool_int):
    test_str = test_str[1:len(test_str)-1]
    res = []
    temp = []
#    print(test_str)
    for token in test_str.split(", "):
        if token:
            if bool_int:
                num = int(token.replace("(", "").replace(")", "").replace("[", "").replace("]", ""))
            else:
                num = str(token.replace("(", "").replace(")", "").replace("'","").replace("[", "").replace("]", ""))
                
            temp.append(num)
            if ")" in token or "]" in token:
               res.append(tuple(temp))
               temp = []
    return res

''' judge if two list has shared elements  '''
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3
def overlap(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2]
    if lst3 and lst3 != lst1 and lst3 != lst2:
        return True
    else:
        return False

''' judge if s is the sub list of l ''' 
def is_Sublist(l, s):
    sub_set = False
    if s == []:
        sub_set = True
    elif s == l:
        sub_set = True
    elif len(s) > len(l):
        sub_set = False
    else:
        for i in range(len(l)):
            if len(l)-i < len(s):
                return False
            if l[i] == s[0]:
                n = 1
                while (n < len(s)) and (l[i+n] == s[n]):
                    n += 1
                if n == len(s):
                    sub_set = True
                    return sub_set
    return sub_set


''' get the TP and FP value given the ground truth and predicted result'''
def interset_num(list1,list2,detected_place_names,place_names,  \
                 ignored_place_names, amb_place_offset=[], amb_place_names=[], hashtag_offsets=[]):
    TP = 0
    FP = 0
    FN_s = 0
    FN_list = [1]*len(place_names)
    place_detect_score = [0]*len(place_names)
    for i, l1 in enumerate(list1):
        lower_place = tuple([item.lower() for item in detected_place_names[i]])
        bool_ins = False
        bool_extra = False
        if lower_place in ignored_place_names:
            for j, l in enumerate(list2):
                if (l1[0] >= l[0] and l1[0] <= l[1]) or (l1[1] >= l[0] and l1[1] <= l[1]) \
                or (l[0] >= l1[0] and l[0] <= l1[1]) or (l[1] >= l1[0] and l[1] <= l1[1]):
                    bool_extra = True
                    break
        if not bool_extra and (lower_place in ignored_place_names):
            TP += 1
            bool_ins = True
        else:
            for j, l in enumerate(list2):
                if l1 == l:
                    place_detect_score[j] = 1  
                    TP += 1
                    FN_list[j] = 0
                    bool_ins = True
                else:
                    if (l1[0] >= l[0] and l1[0] <= l[1]) or (l1[1] >= l[0] and l1[1] <= l[1]) or (l[0] >= l1[0] and l[0] <= l1[1]) or (l[1] >= l1[0] and l[1] <= l1[1]):
                        # import pdb
                        # pdb.set_trace()
                        pen = interset_adv(list(detected_place_names[i]),list(place_names[j]))
                        if pen != 1:
                            pen = 0.5
                        else:
                            TP += pen
                        FP += (1-pen)
                        FN_list[j] = 0
                        if pen != 1:
                            FN_s += 0.5
                        place_detect_score[j] = pen  
                        bool_ins = True
        if not bool_ins:
            bool_changed = 0
            for j, l in enumerate(amb_place_offset):
                if l1 == l:
                    FP += 0
                    TP += 1
                    bool_changed = 1
                    break
                else:
                    if (l1[0] >= l[0] and l1[0] <= l[1]) or (l1[1] >= l[0] and l1[1] <= l[1]) or (l[0] >= l1[0] and l[0] <= l1[1]) or (l[1] >= l1[0] and l[1] <= l1[1]):
                        pen = interset_adv(list(detected_place_names[i]),list(amb_place_names[j]))
                        if pen == 1:
                            FP += 0
                            TP += 1
                        else:
                            FP += 1
                        bool_changed = 1
                        break
            if not bool_changed:
                for hashtag in hashtag_offsets:
                    if (l1[0] >= hashtag[0]+ 1 and l1[1] <= hashtag[1]) and hashtag[1]-(hashtag[0]+ 1) > l1[1] - l1[0]:
                        bool_changed = 1
                        FP += 0
                        break

            if not bool_changed:
                FP += 1
    return TP,FP,sum(FN_list)+FN_s, place_detect_score

def extract_place(pd_item):
    return_places = []
    for item in pd_item:
        if '-' not in item and '/' not in item and '(' not in item:
            item = unicodedata.normalize('NFKD', item).encode('ascii','ignore').decode("utf-8") 
            row_nobrackets = re.sub("[\(\[].:;*?[\)\]]", "", item)
            corpus = [word.lower() for word in re.split("[. #,&\"\',’]",row_nobrackets)]
            new_corpus = []
            for cor in corpus:
            	all_ascii = ''.join(char for char in cor if ord(char) < 127)
            	new_corpus.append(all_ascii)
            corpus = [x for x in new_corpus if x and (len(x) < 2 or (len(x)>=2 and not (x[0]== '(' and x[len(x)-1]== ')')))]
            if corpus:
                return_places.append(tuple(corpus))
    return return_places

def extract_tokens(pos_f):
    very_fre_words = []
    with codecs.open(pos_f, 'r',encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = re.split("\s",line)
            tokens = list(filter(None, tokens))
            tokens = [x.lower() for x in tokens if x]
            very_fre_words.extend(tokens)
    return list(set(very_fre_words))


'''replace the number of a string by 0, such as hwy12 to hwy00'''
def replace_digs(word):
    new_word = ''
    for i, c in enumerate(word):
        if c.isdigit():
            new_word+='0'
        else:
            new_word+=c
    return new_word

def abbrevison1(abr_file):
    with open(abr_file,mode='r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        abbr = {}
        for row in reader:
            tokens = row[0].split(' ')
#            if len(tokens) == 1:
            abbr[row[1]] = tokens 
    return abbr

def abbrevison(abr_file):
    with open(abr_file,mode='r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        abbr = {}
        for row in reader:
            tokens = row[0].split(' ')
            if len(tokens) == 1:
                abbr[row[1]] = tokens[0] 
    return abbr

def mention_locations(target_string):
    mention_offsets = []
    for match in re.finditer(r'@([a-zA-Z0-9]{1,15})', target_string):
        mention_offsets.append((match.start(),match.end()-1))
    return mention_offsets

'''to judege if a string contains numbers'''
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

'''write list of place names into file'''
def write_place(file_name, place_names):
    f= codecs.open(file_name,"w+")
    for neg in place_names:
        temp= ''
        for negraw in neg:
            temp = temp+negraw+' ' 
        f.write(temp+'\n')
    f.close()

def load_f_data(pos_f, very_fre_count):
    pos_training_data = {}
    count = 0
    very_fre_words = []
    with codecs.open(pos_f, 'r',encoding='utf-8') as file:
        for line in file:
            count += 1
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = re.split("\s",line)
            tokens = list(filter(None, tokens))
            tokens = [x for x in tokens if x]
            pos_training_data[tokens[1].lower()] = int(tokens[3])
            if len(very_fre_words) < very_fre_count:
                very_fre_words.append(tokens[1])
    return pos_training_data,very_fre_words

'''load place names from a file'''    
def load_osm_names_fre(pos_f, fre_words, aug_count = 1, return_gen = 0):
    pos_training_data = []
    general_places = set()
    with codecs.open(pos_f, 'r',encoding='utf-8') as file:
        for line in file:
            line = unicodedata.normalize('NFKD', line).encode('ascii','ignore').decode("utf-8") 
            line = line.strip()
            if len(line) == 0:
                continue
            row_nobrackets = re.sub("[\(\[].:;*?[\)\]]", "", line)         
            corpus = [word.lower() for word in re.split("[. #,&\"\',’]",row_nobrackets)]
            corpus = [word  for word in corpus if word]
            corpus = [replace_digs(word) for word in corpus]
            final_result = []
            for token in corpus:
                groups = re.split('(\d+)',token)
                groups = [g for g in groups if g]
                final_result.extend(groups)
            if not(len(final_result) == 1 and final_result[0] in fre_words):
                for k in range(aug_count):
                    pos_training_data.append(tuple(final_result))
            else:
                if return_gen:
                    general_places.add(tuple(final_result))
    if return_gen:
        return pos_training_data, general_places
    else:
        return pos_training_data

def load_osm_names_fre_set(pos_f, fre_words, aug_count = 1, return_gen = 0):
    pos_training_data = set()
    general_places = set()
    with codecs.open(pos_f, 'r',encoding='utf-8') as file:
        for line in file:
            line = unicodedata.normalize('NFKD', line).encode('ascii','ignore').decode("utf-8") 
            line = line.strip()
            if len(line) == 0:
                continue
            row_nobrackets = re.sub("[\(\[].:;*?[\)\]]", "", line)         
            corpus = [word.lower() for word in re.split("[. #,&\"\',’]",row_nobrackets)]
            corpus = [word  for word in corpus if word]
            corpus = [replace_digs(word) for word in corpus]
            final_result = []
            for token in corpus:
                groups = re.split('(\d+)',token)
                groups = [g for g in groups if g]
                final_result.extend(groups)
            if not(len(final_result) == 1 and final_result[0] in fre_words):
                for k in range(aug_count):
                    pos_training_data.add(tuple(final_result))
            else:
                if return_gen:
                    general_places.add(tuple(final_result))
    if return_gen:
        return pos_training_data, general_places
    else:
        return pos_training_data

'''load place names from a file'''    
def load_osm_names_fre1(pos_f, fre_words, aug_count = 1, return_gen = 0):
    pos_training_data = []
    general_places = set()
    with codecs.open(pos_f, 'r',encoding='utf-8') as file:
        for line in file:
            # line = unicodedata.normalize('NFKD', line).encode('ascii','ignore').decode("utf-8") 
            # line = line.strip()
            corpus = line.split()
            if len(corpus) == 0:
                continue
            # row_nobrackets = re.sub("[\(\[].:;*?[\)\]]", "", line)         
            # corpus = [word.lower() for word in re.split("[. #,&\"\',’]",row_nobrackets)]
            # corpus = [word  for word in corpus if word]
            # corpus = [replace_digs(word) for word in corpus]
            pos_training_data.append(tuple(corpus))
    return pos_training_data


def load_word_index(index_file):
    word2idx = {}
    max_char_len = 0
    with open(index_file, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            if len(word) > max_char_len:
                max_char_len = len(word)
            if len(line)== 1:
                # print('errors in vocab list')
                # print(line)
                # print(len(word2idx))
                word2idx['jiyougguexjcgsnoword'] = int(line[0])
            else:
                if line[0] in word2idx.keys():
                    # print('appear before:',line[0])
                    word2idx[line[0]+str(len(word2idx))] = int(line[1])
                else:
                    word2idx[line[0]] = int(line[1])
    return word2idx, max_char_len

'''load place names from a file'''    
def string2pla(rawstr):
    line = rawstr.strip()
    row_nobrackets = re.sub("[\(\[].:;*?[\)\]]", "", line)         
    corpus = [word.lower() for word in re.split("[. #,&\"\',’]",row_nobrackets)]
    corpus = [word  for word in corpus if word]
#    tokens = line.split(' ')
    return tuple(corpus)

def load_osm_names(pos_f):
    pos_training_data = []
    with codecs.open(pos_f, 'r',encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split(' ')
            pos_training_data.append(tuple(tokens[0:len(tokens)])) 
    return pos_training_data

isascii = lambda s: len(s) == len(s.encode())

'''split a word to multiple sub words by numbers'''    
def split_numbers(word):
    groups = re.split('(\d+)',word)
    num_tag = False
    front_word = ''
    back_word = ''
    for g in groups:
        if hasNumbers(g):
            num_tag = True
        else:
            new_word = "".join(re.findall("[a-zA-Z]*", g))
            if new_word:
                if num_tag :
                    back_word = new_word
                    break
                else:
                    front_word = new_word

    return front_word, back_word
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

if __name__ == '__main__':
#    read_ent_file('data/word_ent100_copy.txt')
#    print(extract_tokens('data/fc.txt'))
#    print(index_list_int(1,[0,2,3],[[0,1,2],[0,1,3],[0,1,2,3,4,],[0,1]]))
#    print(index_list_sub(1,[0,2,3],[[0,2],[0,1,3],[0,1,3,4],[0,1]]))
#    print(using_split2('Comic-con','[^a-zA-Z^0-9]'))
    bigram = load_bigram_model('model/1004232731-bigram.txt')
    save_bigram_model(bigram,'data/re.txt')
    #    print(string2pla('us flood.'))
#    test = '1233fc55tg- f'
#    print(split_numbers(test))
#    print(re.findall(r'[A-Za-z]|-?\d+\.\d+|\d+',test))
#    res = [re.findall(r'(\w+?)(\d+)', test)[0] ]
#    groups = re.split('(\d+)',test)
#    print(replace_digs('5578sfhfhjf22'))
#    print(softmax([0.91,0.03,0.05,0.01]))
#    item= "RT @iH8TvvitterHoes: Nigga that's Nuketown Rtì@HistoryInPix: The Great Alaska Earthquake of 1964 http://t.co/CGQzLUahHUî"
#    item = unicodedata.normalize('NFKD', item).encode('ascii','ignore').decode("utf-8") 
#    print(item)
#if __name__ == '__main__':
#    # main()
#    print(softmax([0.91,0.03,0.05,0.01]))
