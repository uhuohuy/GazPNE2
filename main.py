#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 17:03:40 2020

@author: hu_xk
"""
import os
import sys
from datetime import datetime
from place_tagger import place_tagging
from utility import *
import argparse
#sys.path.append(os.path.abspath('unsupervised_NER'))
import time
from main_NER import UnsupNER

def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--id', type=str, default='0622143005')
    parser.add_argument('--osmembed', type=int, default= 7)
    parser.add_argument('--thres1', type=float, default= 0.7)
    parser.add_argument('--filter_l', type=int, default= 1)
    parser.add_argument('--bool_osm', type=int, default= 0)
    parser.add_argument('--emb', type=int, default= 1)
    parser.add_argument('--cnn', type=int, default= 150)
    parser.add_argument('--lstm', type=int, default= 150)
    parser.add_argument('--special_con_t', type=float, default= 0.35)
    parser.add_argument('--input', type=int, default= 4)
    parser.add_argument('--input_file', type=str, default= 'test.txt')
    parser.add_argument('--epoch', type=int, default= 4)
    parser.add_argument('--abb_ent_thres', type=float, default= 0.3)
    parser.add_argument('--context_thres', type=float, default= 0.3)
    parser.add_argument('--abb_context_thres', type=float, default= 0.2)
    parser.add_argument('--num_context_thres', type=float, default= 0.2)
    parser.add_argument('--single_person_c_t', type=float, default= 0.23)
    parser.add_argument('--osm', type=int, default= 1)
    parser.add_argument('--weight', type=int, default= 1)
    parser.add_argument('--bool_fast', type=int, default= 1)
    parser.add_argument('--special_ent_t', type=float, default= 0.3)
    parser.add_argument('--bool_general_check', type=int, default= 1)
    parser.add_argument('--general_words', type=int, default= 26000)
    parser.add_argument('--merge_thres', type=float, default= 0.5)
    parser.add_argument('--dic_neig', type=int, default= 301)
    parser.add_argument('--con_neig', type=int, default= 301)
    parser.add_argument('--emw', type=int, default= 4)
    parser.add_argument('--fc_ratio', type=float, default= 0.25)
    parser.add_argument('--bool_debug', type=int, default= 0)

    args = parser.parse_args()
    
    print ('id: '+str(args.id))
    print ('thres: '+str(args.thres1))
    print ('filter_l: '+str(args.filter_l))
    print ('emb: '+str(args.emb))
    print ('cnn: '+str(args.cnn))
    print ('lstm: '+str(args.lstm))
    print ('special_con_t: '+str(args.special_con_t))
    print ('input: '+str(args.input))
    print ('input_file: '+str(args.input_file))
    print ('epoch: '+str(args.epoch))
    print ('abb_ent_thres: '+str(args.abb_ent_thres))
    print ('context_thres: '+str(args.context_thres))
    print ('abb_context_thres: '+str(args.abb_context_thres))
    print ('num_context_thres: '+str(args.num_context_thres))
    print ('single_person_c_t: '+str(args.single_person_c_t))

    print ('osm: '+str(args.osm))
    print ('weight: '+str(args.weight))
    print ('bool_fast: '+str(args.bool_fast))
    print ('special_ent_t: '+str(args.special_ent_t))
    print ('bool_general_check: '+str(args.bool_general_check))
    print ('general_words: '+str(args.general_words))
    print ('merge_thres: '+str(args.merge_thres))
    print ('dic_neig: '+str(args.dic_neig))
    print ('con_neig: '+str(args.con_neig))
    print ('emw: '+str(args.emw))
    print ('fc_ratio: '+str(args.fc_ratio))
    print ('bool_debug: '+str(args.bool_debug))
    
    start_time = time.time()

    obj = UnsupNER(args.dic_neig,args.con_neig,args.emw)
    print('UnsupNER', time.time()-start_time)

    time_str = datetime.now().strftime('%m%d%H%M%S')
    print('time_str',time_str)
        
    if args.input == 2:
        regions=[32,31,30]
    elif args.input == 0:
        regions=[49]
    elif args.input == 5:
        regions=[4]
    elif args.input == 6:
        regions=[50]
    elif args.input == 7:
        regions=[51]
    elif args.input == 8:
        regions=[52]
    elif args.input == 9:
        regions=[53]
    elif args.input == 10:
        regions=[55]
    elif args.input == 11:
        regions=[56]
    elif args.input == 12:
        regions=[57]
    elif args.input == 13:
        regions=[58]
    elif args.input == 14:
        regions=[59]
    else:
        regions = [0,25,26,6,7, 20, 21, 9,10,28,17, 18, 19, 8,15,1,2,11,12,13,14]
    fc_file='data/fc.txt'
    fc_tokens = extract_tokens(fc_file)
    file_name = 'data/osm_abbreviations_globe.csv'
        
    sim_abv = abbrevison1(file_name)
    abv_punk = {}
    for key in sim_abv.keys():
        new_abv = ''
        for i, char in enumerate(key):
                new_abv += char + '.'
        abv_punk[new_abv]=key
    #load the osm place names
    start_time = time.time()

    if args.osm:
#        if args.F1 == 4:
        osm_names = load_osm_names_fre('data/'+str(args.id)+str(args.epoch)+'.txt', [], aug_count = 1)
        osm_names = [item for item in osm_names if len(item) > 1]
        osm_names = set(osm_names)
#        else:
#            osm_names = load_osm_names_fre('data/country.txt', [], aug_count = 1)
#            osm_names = set(osm_names)
    else:
        osm_names = []
    print('osm_names', time.time()-start_time)
    start_time = time.time()
        
    if args.bool_general_check:
        general_file = 'data/general_place'+str(args.general_words)+'.txt'
        if os.path.isfile(general_file):
            general_place_list = load_osm_names_fre(general_file, [], aug_count = 1)
        else:
            general_place_list = []
        candidate_file = 'data/candidates'+str(args.general_words)+'.txt'
        if os.path.isfile(candidate_file):
            candidate_words = load_osm_names_fre(candidate_file, [], aug_count = 1)
            general_place_list.extend(candidate_words)
    else:
        general_place_list = []
    print('load_osm_names_fre', time.time()-start_time)
        
    # Write each dataframe to a different worksheet.
    for file in os.listdir("model/"):
        if args.id in file and '.pkl' in file:
            if 'clstm' in file:
                try:
                    new_name = 'model/'+file[12:len(file)]
                    os.rename('model/'+file, new_name)
                    epoch_str = file[27:len(file)][:-4]
                    epoch = int(epoch_str)
                except BaseException:
                    continue
            else:
                epoch_str = file[15:len(file)][:-4]
                epoch = int(epoch_str)
            print('epoch:'+str(epoch))
            if epoch == args.epoch:
                for r_idx, region in enumerate(regions):
                    F1, P,R = place_tagging(int(args.input == 1), time_str,obj,args.thres1,args.id,args.osmembed,args.bool_osm,1,args.cnn,region,\
                                args.lstm,epoch,args.filter_l,1,osm_names, args.emb, \
                                args.special_con_t, args.abb_ent_thres, args.context_thres, \
                                                               args.weight,args.bool_fast, args.special_ent_t, \
                                                               general_place_list,abv_punk,\
                                                               args.merge_thres,fc_tokens,args.fc_ratio,args.input_file,\
                                                               args.abb_context_thres, args.num_context_thres, args.single_person_c_t,args.bool_debug)
                    print('region '+ str(region)+' : F1 ' + str(F1)+' : thres ' + str(args.thres1))
#    writer.save()
if __name__ == '__main__':
    main()
