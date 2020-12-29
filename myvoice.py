#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Created on Sunday October 2, 2016 @ 23:32:39  

@author: Tanmay Basu
@description: Qualitative Text Analysis in Healthcare using NLP  
""" 

import csv,re,sys
from nltk import pos_tag,PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

class qualitative_text_analysis():
    def __init__(self,path='/home/project/what_qualities.txt',metric='w',pos='n',transformation='d'):
        self.path = path
        self.metric=metric
        self.pos=pos
        self.transformation=transformation

    # Common subsequence 
    def common_subsequence(self,w1,w2):
        if len(w1)<len(w2):
            return self.common_subsequence(w2,w1)
        else:
            cp=[]
            for i in range(0,len(w2)):      # smaller word 
                if w1[i]==w2[i]:
                    cp.append(w1[i])
                else:
                    return cp
        return cp
    
    # Extract derivationally related form and pertainyms 
    def extract_derivational_forms(self,sns,w,lst):
        lm=sns.lemmas() 
        for e in lm:
            drf=e.derivationally_related_forms()                         # derivationally related forms 
            if drf!=[]:
                for elm in drf:
                    match=re.search(r'((\w)+(\.))+(\w)+',str(elm))       # storing only the first derivational form 
                    if match:
                        dw=match.group(0).split('.')[0]
                        cs=self.common_subsequence(w,dw)
                        if len(cs)>=len(w)*0.5 or len(cs)>=len(dw)*0.5: 
                            lst.append(dw)  
        return lst    
                
    # Merge two synonyms 
    def merging_synsets(self,syns,w1,w2,sns1,sns2,max_score,loc):
        syns[loc].insert(0,max_score)
        syns[loc].insert(1,w1)
        syns[loc].insert(2,sns1.name())
        syns[loc].insert(3,w2)
        syns[loc].insert(4,sns2.name()) 
        if self.transformation=='d':
            lst=[]
            lst=self.extract_derivational_forms(sns1,w1,lst)
            lst=self.extract_derivational_forms(sns2,w2,lst) 
            #syns[loc]=list(sorted(set(syns[loc]),key=syns[loc].index))
            for e in lst:
                syns[loc].append(e) 
        loc=loc+1
        return syns,loc
        
    # Finding most similar concept between two words 
    def word_similarity(self,w1,w2,syns,loc,thr_sim):
        syn1=wn.synsets(w1, wn.NOUN or wn.ADJ)    
        syn2=wn.synsets(w2, wn.NOUN or wn.ADJ)    
    
        if len(syn1)>0 and len(syn2)>0:
            score=0; max_score=0; count=0
            sns1=syn1[0]
            sns2=syn2[0] 
            for i in range (0,len(syn1)):
                for j in range (0,len(syn2)):
                    if self.metric=='j':                                            # Jiang-Conrath Similarity
                        score=wn.jcn_similarity(syn1[i],syn2[j])
                    elif self.metric=='le':                                         # Leacock-Chodorow Similarity 
                        score=wn.lch_similarity(syn1[i],syn2[j],simulate_root=False) 
                    elif self.metric=='li':                                         # Lin Similarity
                        score=wn.lin_similarity(syn1[i],syn2[j])
                    elif self.metric=='p':                                          # Path Similarity 
                        score=wn.path_similarity(syn1[i],syn2[j])
                    elif self.metric=='w':                                          # Wu-Palmer Similarity. It can not be '0'. It ranges in (0,1] 
                        score=wn.wup_similarity(syn1[i],syn2[j])  
                        
                    if score>max_score:                         # Finding the maximum score              
                        max_score=score
                        sns1=syn1[i] 
                        sns2=syn2[j] 
                        if max_score>=thr_sim:                     # Storing all the synset pairs that have scores > threshold 
                            syns,loc=self.merging_synsets(syns,w1,w2,sns1,sns2,max_score,loc)                       
                            count=count+1 
            if count==0:                           # Storing the synset that has maximum score but the score < threshold
                syns,loc=self.merging_synsets(syns,w1,w2,sns1,sns2,max_score,loc)   
        return syns,loc 
    
    # Clustering of term sets
    def clustering_terms(self,processed_terms,loc,thr_merge,flag): 
        for i in range(0,loc-1):                    # Finding intersection between two sets to merge them, if necessary 
            if processed_terms[i][0]!='':
                syn1=[x for x in processed_terms[i] if x.find('.')!=-1]            # First cluster 
                for j in range(i+1,loc): 
                     if processed_terms[j][0]!='':
                         syn2=[x for x in processed_terms[j] if x.find('.')!=-1]   # Second cluster 
                         common_syn=[x for x in processed_terms[i] if x.find('.')!=-1 and x in processed_terms[j]]    # Intersection of two clusters               
                         prcnt=(float(len(common_syn))/(len(set(syn1+syn2))))*100   # % of intersection 
                         if prcnt>=thr_merge:
                             flag=1   
                             for k in range(0,len(processed_terms[j])):                 # merging common sets of words 
                                processed_terms[i].append(str(processed_terms[j][k]))
                                processed_terms[j][k]=''                             
        if flag==1:
            return self.clustering_terms(processed_terms,loc,thr_merge,0)    # Iteration 
        else:                   
            return processed_terms
    
    # Remove duplicates and find frequency of all unique terms in a cluster 
    def process_cluster(self,data,all_terms,unique_terms,lst,rank):
    #    lst=list(sorted(set(lst),key=lst.index))
        tmp=[]
        for e in lst:               # Removing duplicates 
            if e.find('.')!=-1:
                tmp.append(e) 
            else:
                if e not in tmp:  
                    tmp.append(e)
        lst=tmp
        cn=0; new_lst=[]; sent=''; 
        for elm in lst:                          
            cn=cn+all_terms.count(elm)                             # Finding the frquencies of the terms
            for i in range(0,len(data)):
                bs=word_tokenize(str(data[i])) 
                if bs.count(elm)>0:                                # Finding the sentences that contains the token
                    if len(bs)>4:                                  # Displaying the first and last two terms of such sentences
                        ln=len(bs)                        
                        sent=sent+'\n'+str(i+2)+', '+str(bs[0])+' '+str(bs[1])+' ... '+str(bs[ln-2])+' '+str(bs[ln-1])
                    else:
                        sent=sent+'\n'+str(i+2)+', '+str(data[i]) 
            if unique_terms.count(elm)>0:
                unique_terms[unique_terms.index(elm)]='' 
                
        new_lst.append(sent)                                       # Print the sentences where the terms in a cluster appear
        new_lst.append(rank)                                       # Rank of the cluster according to wordnet similarity 
        new_lst.append(cn)
        for elm in lst:
            new_lst.append(elm)  
        return unique_terms,new_lst  
    
    # Loading the clusters into files aftre mapping each word to its actual form 
    def store_cluster(self,data,path,clusters,loc,all_terms,unique_terms,stems,all_processed_terms):
        flr = open(path, 'w') 
        wr = csv.writer(flr, delimiter=',',dialect='excel')  
        word_clusters=[] 
        count=0; rank=1
        for rw in clusters[0:loc]: 
            if rw[0]!='':
                lst=[];
                if self.transformation=='s':                       # Mapping the stems to the roots     
                    for i in range(0,len(rw)):                
                        for j in range (0,len(stems)): 
                            if rw[i]==stems[j]:
                                lst.append(all_processed_terms[j]) 
                else:                                  # If stemming is not used 
                    for i in range(0,len(rw)):
                        lst.append(rw[i])
                count=count+1 
                unique_terms,new_lst=self.process_cluster(data,all_terms,unique_terms,lst,rank) 
                rank=rank+1
                word_clusters.append(new_lst)
    #            wr.writerow(new_lst)                    
        for elm in unique_terms:                        # Finding the frquencies of the single terms 
            if elm!='':
                lst=[] 
                lst.append(elm) 
                if self.transformation=='d': 
                    if self.pos=='a':                               # Only adjectives
                        syn=wn.synsets(elm, wn.ADJ)
                    elif self.pos=='n':                             # Only nouns
                        syn=wn.synsets(elm, wn.NOUN)    
                    elif self.pos=='b':                             # Both adjectives and nouns
                        syn=wn.synsets(elm, wn.NOUN or wn.ADJ) 

                    if len(syn)>0:
                        for i in range(0,len(syn)):
                            lst=self.extract_derivational_forms(syn[i],elm,lst)  
                unique_terms,new_lst=self.process_cluster(data,all_terms,unique_terms,lst,rank) 
                word_clusters.append(new_lst)
    #            wr.writerow(new_lst)   
        word_clusters=sorted(word_clusters, key = lambda x: x[2], reverse=True)
        heading=[]
        heading.append('Row No')
        heading.append('Rank by Wup Similarity')
        heading.append('Frequency')
        wr.writerow(heading)
        for rw in word_clusters:
            wr.writerow(rw)                 
        flr.close()  
        return count
    
    # The main function   
    def get_summary(self):

        outfile1='./'+self.path.split('/')[-1].strip('.txt')+'_synonyms.csv'
        outfile2='./'+self.path.split('/')[-1].strip('.txt')+'_summary.csv'    
        fl=open(self.path, 'r')
        text=list(csv.reader(fl,delimiter='\n'))
        fl.close()    
        sentences=[item for sublist in text for item in sublist]
    
        data=[]; nouns=[]; adjectives=[]; all_terms=[] 
        for sentence in sentences:  
            data.append(sentence)
            for w,pos in pos_tag(word_tokenize(sentence)):
                all_terms.append(w.lower()) 
                if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                     nouns.append(w.lower())
                elif (pos == 'JJ' or pos == 'JJR' or pos == 'JJS'):
                     adjectives.append(w.lower())
                     
        data=[x.lower() for x in data] 
        thr_merge=50                            # Threshold on percentage of intersection between two clusters    
        if self.metric=='j':
            thr_sim=0.9                         # Jiang-Conrath similarity threshold 
        elif self.metric=='le':
            thr_sim=2.8                         # Leacock-Chodorow (LCH) similarity threshold 
        elif self.metric=='li':
            thr_sim=0.9                         # Lin similarity threshold 
        elif self.metric=='p':
            thr_sim=0.4                         # Path similarity threshold
        elif self.metric=='w':
            thr_sim=0.9                         # Wu-Palmer similarity threshold 
        else:
            print('\n ##### Wrong input for wordnet metric. Enter correctly. ##### \n\n The options are '
                  '\n\t "j" for Jiang-Conrath similarity,'
                  '\n\t "le" for Leacock-Chodorow (LCH) similarity'
                  '\n\t "li" for Lin similarity'
                  '\n\t "li" for Path similarity'
                  '\n\t "li" for Wu-Palmer similarity \n\n')
            sys.exit(0)
            
        if self.pos=='a':                   # Process adjectives only
            terms=adjectives    
        elif self.pos=='n':                 # Process nouns only
            terms=nouns
        elif self.pos=='b':                 # Process both adjectives and nouns
            terms=adjectives+nouns
        else:
            print('\n ##### Wrong input for POS. Enter correctly. ##### \n\n The options are '
                  '\n\t "a" for adjectives,'
                  '\n\t "b" for both adjectives and nouns'
                  '\n\t "n" for nouns \n\n')
            sys.exit(0)    
    # Synsets generation   
        loc=0
        unique_terms=list(set(terms))
    #    print unique_terms 
        ln=len(unique_terms)
        ln=ln*(ln-1)/2 
        syns = [[]for x in range(int(ln))]     
        for i in range(0,len(unique_terms)-1):
            for j in range(i+1,len(unique_terms)):
                syns,loc=self.word_similarity(unique_terms[i],unique_terms[j],syns,loc,thr_sim) # Finding most similar synset 
              
        syns=sorted(syns[0:loc], key = lambda x: x[0], reverse=True)   # Sort the synsets in decreasing order of scores
    #    print syns[0] 
        heading = []
        heading.append('Score')
        heading.append('Term1')
        heading.append('Term2')
        heading.append('Synset1')
        heading.append('Synset2')        
        processed_terms = [[]for x in range(int(ln))]
        loc=0
        flr = open(outfile1, 'w')
        wr = csv.writer(flr, delimiter=',',dialect='excel')     # Writing the synsets to a file along with the words 
        wr.writerow(heading)
    #   Writing the words with their synsets
        stems=[]; all_processed_terms=[] 
        for rw in syns:
            wr.writerow(rw) 
            if rw[0]>=thr_sim:                      
                for i in range(1,len(rw)): 
                    if self.transformation=='s':                                   # Perform stemming
                        stm=PorterStemmer().stem_word(rw[i])             
                        if all_processed_terms.count(rw[i])==0:
                            stems.append(stm) 
                            all_processed_terms.append(rw[i]) 
                        processed_terms[loc].append(str(stm))
                    else:
                        processed_terms[loc].append(str(rw[i]))   # Consider actual words                    
                loc=loc+1
        flr.close()
    
        clusters=self.clustering_terms(processed_terms,loc,thr_merge,0)         # Clustering 
        count=self.store_cluster(data,outfile2,clusters,loc,all_terms,unique_terms,stems,all_processed_terms) 
        print('No. of Clusters: '+ str(count))
