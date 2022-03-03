#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:37:05 2021

@author: hu_xk
"""
from utility import *


# def fusion_strategy28(ent_prob,context_ent_prob, ent_thres, context_thres, entity, abbr, \
#                       bool_general, pos_lists, abb_context_thres = 0.15, merge_thres=0.35, \
#                       abb_ex_t = 0.4, num_context_thres = 0.1, add_prob = 0, context_thres_gene = 0.3, single_person_c_t = 0.2):
#     if not bool_general:
#         disturb = 0.000001
#     #    add_prob = 0.25
#         contain_number = 0
#         for item in entity:
#             if hasNumbers(item):
#                 contain_number = 1
#                 break
#         if contain_number:
#             bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))

#             if bool_context_loc and context_ent_prob['LOC'] >= num_context_thres:
#                 return True
#             else:
#                 return False
#         else:
#             if len(entity) == 1 and (entity[0].lower() in abbr or 3 > len(entity[0])): # 
#                 if pos_lists[0] in PERSON_POS_ADV and entity[0].lower() in abbr : 
#                     bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
#                     if (bool_context_loc and context_ent_prob['LOC'] > abb_context_thres) or ent_prob['LOC'] > ent_thres:
#                         return True
#                     else:
#                         return False
    
#                 else:
#                    bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
#     #                    
#                    if context_ent_prob['LOC'] >= abb_context_thres + abb_ex_t and bool_context_loc:
#                        return True
#                    else:
#                         return False
#             else:
#                 merged_ent_prob = {k: ent_prob.get(k, 0) + context_ent_prob.get(k, 0) \
#                            for k in set(ent_prob) | set(context_ent_prob)}
#             #    print('context_ent_prob', context_ent_prob)
#                 merged_ent_prob['LOC']+=disturb
#                 bool_total_loc = int('LOC' == max(merged_ent_prob, key=merged_ent_prob.get))
#                 context_ent_prob['LOC']+=disturb
#                 bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
                
#                 if not ('PER' in context_ent_prob.keys() and context_ent_prob['PER'] > single_person_c_t and \
#                     context_ent_prob['PER'] > context_ent_prob['LOC']) and ( (merged_ent_prob['LOC'] >= merge_thres  and bool_total_loc)  \
#                      or (context_ent_prob['LOC'] >= context_thres and bool_context_loc)): # ent_thres > ent_prob['LOC']  and 
#                     return True
#                 else:
#                     return False

#     else:
#         bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
#         if bool_context_loc and context_ent_prob['LOC'] >= context_thres_gene: # ent_thres > ent_prob['LOC']  and 
#             return True
#         else:
#             return False

# def fusion_strategy30(ent_prob,context_ent_prob, ent_thres, context_thres, entity, abbr, \
#                       bool_general, pos_lists, abb_context_thres = 0.2, merge_thres=0.5, \
#                      num_context_thres = 0.2,  single_person_c_t = 0.23,  abb_ex_t = 0.4, \
#                      context_thres_gene = 0.7):
#     if not bool_general:
#         disturb = 0.000001
#     #    add_prob = 0.25
#         contain_number = 0
#         for item in entity:
#             if hasNumbers(item):
#                 contain_number = 1
#                 break
#         if contain_number:
#             bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))

#             if bool_context_loc and context_ent_prob['LOC'] >= num_context_thres:
#                 return True
#             else:
#                 return False
#         else:
#             if len(entity) == 1 and (entity[0].lower() in abbr or 4 > len(entity[0])): # 
#                 if pos_lists[0] in PERSON_POS_ADV and entity[0].lower() in abbr: 
#                     bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
#                     if (bool_context_loc and context_ent_prob['LOC'] > abb_context_thres) or ent_prob['LOC'] > ent_thres:
#                         return True
#                     else:
#                         return False
#                 else:
#                    bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
#     #                    
#                    if context_ent_prob['LOC'] >= abb_context_thres + abb_ex_t and bool_context_loc:
#                        return True
#                    else:
#                         return False
#             else:
#                 merged_ent_prob = {k: ent_prob.get(k, 0) + context_ent_prob.get(k, 0) \
#                            for k in set(ent_prob) | set(context_ent_prob)}
#             #    print('context_ent_prob', context_ent_prob)
#                 merged_ent_prob['LOC']+=disturb
#                 bool_total_loc = int('LOC' == max(merged_ent_prob, key=merged_ent_prob.get))
#                 context_ent_prob['LOC']+=disturb
#                 bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
#                 if len(entity) > 1:
#                     context_reduced = 0.12
#                 else:
#                     context_reduced = 0
#                 if not ('PER' in context_ent_prob.keys() and context_ent_prob['PER'] > single_person_c_t and \
#                     context_ent_prob['PER'] > context_ent_prob['LOC']) and ( (merged_ent_prob['LOC'] >= merge_thres  and bool_total_loc)  \
#                      or (context_ent_prob['LOC'] >= (context_thres-context_reduced) and bool_context_loc)): # ent_thres > ent_prob['LOC']  and 
#                     return True
#                 else:
#                     return False
#     else:
#         bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
#         if bool_context_loc and context_ent_prob['LOC'] >= context_thres_gene: # ent_thres > ent_prob['LOC']  and 
#             return True
#         else:
#             return False



def fusion_strategy29(ent_prob,context_ent_prob, ent_thres, context_thres, entity, abbr, \
                      bool_general, pos_lists, abb_context_thres = 0.2, merge_thres=0.5, \
                     num_context_thres = 0.2,  single_person_c_t = 0.23,  abb_ex_t = 0.4, \
                     context_thres_gene = 0.7,sentence_len=1):
    if not bool_general:
        disturb = 0.000001
    #    add_prob = 0.25
        contain_number = 0
        for item in entity:
            if hasNumbers(item):
                contain_number = 1
                break
        if contain_number:
            bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))

            if bool_context_loc and context_ent_prob['LOC'] >= num_context_thres:
                return True
            else:
                return False
        else:
            if len(entity) == 1 and (entity[0].lower() in abbr or 4 > len(entity[0])): # 
                if pos_lists[0] in PERSON_POS_ADV and entity[0].lower() in abbr : 
                    bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
                    if (bool_context_loc and context_ent_prob['LOC'] > abb_context_thres) or ent_prob['LOC'] > ent_thres:
                        return True
                    else:
                        return False
                else:
                   bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
    #                    
                   if context_ent_prob['LOC'] >= abb_context_thres + abb_ex_t and bool_context_loc:
                       return True
                   else:
                        return False
            else:
                merged_ent_prob = {k: ent_prob.get(k, 0) + context_ent_prob.get(k, 0) \
                           for k in set(ent_prob) | set(context_ent_prob)}
            #    print('context_ent_prob', context_ent_prob)
                merged_ent_prob['LOC']+=disturb
                bool_total_loc = int('LOC' == max(merged_ent_prob, key=merged_ent_prob.get))
                context_ent_prob['LOC']+=disturb
                bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
                
                if not ('PER' in context_ent_prob.keys() and context_ent_prob['PER'] > single_person_c_t and \
                    context_ent_prob['PER'] > context_ent_prob['LOC']) and ( (merged_ent_prob['LOC'] >= merge_thres  and bool_total_loc)  \
                     or (context_ent_prob['LOC'] >= context_thres and bool_context_loc)): # ent_thres > ent_prob['LOC']  and 
                    return True
                else:
                    return False
    else:
        bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
        if bool_context_loc and context_ent_prob['LOC'] >= context_thres_gene: # ent_thres > ent_prob['LOC']  and 
            return True
        else:
            return False

def fusion_strategy30(ent_prob,context_ent_prob, ent_thres, context_thres, entity, abbr, \
                      bool_general, pos_lists, abb_context_thres = 0.2, merge_thres=0.5, \
                     num_context_thres = 0.2,  single_person_c_t = 0.23,  abb_ex_t = 0.4, \
                     context_thres_gene = 0.7,sentence_len=1):
    if not bool_general:
        disturb = 0.000001
    #    add_prob = 0.25
        contain_number = 0
        for item in entity:
            if hasNumbers(item):
                contain_number = 1
                break
        if contain_number:
            bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))

            if bool_context_loc and context_ent_prob['LOC'] >= num_context_thres:
                return True
            else:
                return False
        else:
            if len(entity) == 1 and (entity[0].lower() in abbr or 4 > len(entity[0])): # 
                if pos_lists[0] in PERSON_POS_ADV and entity[0].lower() in abbr : 
                    bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
                    if (bool_context_loc and context_ent_prob['LOC'] > abb_context_thres) or ent_prob['LOC'] > ent_thres:
                        return True
                    else:
                        return False
                else:
                   bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
    #                    
                   if context_ent_prob['LOC'] >= abb_context_thres + abb_ex_t and bool_context_loc:
                       return True
                   else:
                        return False
            else:
                merged_ent_prob = {k: ent_prob.get(k, 0) + context_ent_prob.get(k, 0) \
                           for k in set(ent_prob) | set(context_ent_prob)}
            #    print('context_ent_prob', context_ent_prob)
                merged_ent_prob['LOC']+=disturb
                bool_total_loc = int('LOC' == max(merged_ent_prob, key=merged_ent_prob.get))
                context_ent_prob['LOC']+=disturb
                bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
                
                if not ('PER' in context_ent_prob.keys() and context_ent_prob['PER'] > single_person_c_t and \
                    context_ent_prob['PER'] > context_ent_prob['LOC']) and ( (merged_ent_prob['LOC'] >= merge_thres  and bool_total_loc)  \
                     or (context_ent_prob['LOC'] >= context_thres and bool_context_loc)): # ent_thres > ent_prob['LOC']  and 
                    if sentence_len > 7 and len(entity) == 1:
                        if  context_ent_prob['LOC'] >= 0.1:
                            return True
                        else:
                            print('must_check_false',entity)
                            return False
                    else:
                        return True
                else:
                    return False
    else:
        bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
        if bool_context_loc and context_ent_prob['LOC'] >= context_thres_gene: # ent_thres > ent_prob['LOC']  and 
            return True
        else:
            return False
# def fusion_strategy30(ent_prob,context_ent_prob, ent_thres, context_thres, entity, abbr, \
#                       bool_general, pos_lists, abb_context_thres = 0.2, merge_thres=0.5, \
#                      num_context_thres = 0.2,  single_person_c_t = 0.23,  abb_ex_t = 0.4, \
#                      context_thres_gene = 0.4,output=0):
#     if not intersection(pos_lists,long_places_pos) and len(entity) >= 2:
#         if output:
#             print('pos_error:',pos_lists,entity)
#         return False
#     if not bool_general:
#         disturb = 0.000001
#     #    add_prob = 0.25
#         contain_number = 0
#         for item in entity:
#             if hasNumbers(item):
#                 contain_number = 1
#                 break
#         if contain_number:
#             bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))

#             if bool_context_loc and context_ent_prob['LOC'] >= num_context_thres:
#                 return True
#             else:
#                 return False
#         else:
#             if len(entity) == 1 and (entity[0].lower() in abbr or 4 > len(entity[0])): # 
#                 if pos_lists[0] in PERSON_POS_ADV and entity[0].lower() in abbr : 
#                     bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
#                     if (bool_context_loc and context_ent_prob['LOC'] > abb_context_thres) or ent_prob['LOC'] > ent_thres:
#                         return True
#                     else:
#                         return False
    
#                 else:
#                    bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
#     #                    
#                    if context_ent_prob['LOC'] >= abb_context_thres + abb_ex_t and bool_context_loc:
#                        return True
#                    else:
#                         return False
#             else:
#                 merged_ent_prob = {k: ent_prob.get(k, 0) + context_ent_prob.get(k, 0) \
#                            for k in set(ent_prob) | set(context_ent_prob)}
#             #    print('context_ent_prob', context_ent_prob)
#                 merged_ent_prob['LOC']+=disturb
#                 bool_total_loc = int('LOC' == max(merged_ent_prob, key=merged_ent_prob.get))
#                 context_ent_prob['LOC']+=disturb
#                 bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
                
#                 if not ('PER' in context_ent_prob.keys() and context_ent_prob['PER'] > single_person_c_t and \
#                     context_ent_prob['PER'] > context_ent_prob['LOC']) and ( (merged_ent_prob['LOC'] >= merge_thres  and bool_total_loc)  \
#                      or (context_ent_prob['LOC'] >= context_thres and bool_context_loc)): # ent_thres > ent_prob['LOC']  and 
#                     return True
#                 else:
#                     return False
#     else:
#         bool_context_loc = int('LOC' == max(context_ent_prob, key=context_ent_prob.get))
#         if bool_context_loc and context_ent_prob['LOC'] >= context_thres_gene: # ent_thres > ent_prob['LOC']  and 
#             return True
#         else:
#             return False






def FC_check(fc_tokens,desc,entity,ratio=0.3):
    count=0
    bool_key = 0
    for item in entity:
        if str(item).lower() in fc_tokens:
            bool_key = 1
            break
        
    if bool_key:
        for word in desc:
            if str(word).lower() in fc_tokens:
                count+= 1
    if len(desc):
        if float(count) / float(len(desc))>ratio:
            return True,count
    return False,count
            
    
