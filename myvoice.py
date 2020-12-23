# -*- coding: utf-8 -*- 
""" 
Created on Sun Oct 2 23:32:39 2016 

@author: Tanmay Basu
@description: MyVoice Data Classification  
""" 

import sys,re,csv
from nltk import pos_tag,PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer

# Common subsequence 
def common_subsequence(w1,w2):
    if len(w1)<len(w2):
        return common_subsequence(w2,w1)
    else:
        cp=[]
        for i in range(0,len(w2)):      # smaller word 
            if w1[i]==w2[i]:
                cp.append(w1[i])
            else:
                return cp
    return cp

# Extract derivationally related form and pertainyms 
def extract_derivational_forms(sns,w,lst):
    lm=sns.lemmas() 
    for e in lm:
        drf=e.derivationally_related_forms()                         # derivationally related forms 
        if drf!=[]:
            for elm in drf:
                match=re.search(r'((\w)+(\.))+(\w)+',str(elm))       # storing only the first derivational form 
                if match:
                    dw=match.group(0).split('.')[0]
                    cs=common_subsequence(w,dw)
                    if len(cs)>=len(w)*0.5 or len(cs)>=len(dw)*0.5: 
                        lst.append(dw)  
#        pnm=e.pertainyms()                                          # pertainyms of the lemma     
#        if pnm!=[]:
#            for elm in pnm:
#                match=re.search(r'((\w)+(\.))+(\w)+',str(pnm[0]))   # storing only the first pertainym 
#                if match:
#                   syns[loc].append(match.group(0).split('.')[0])
#                   syns[loc].append(match.group(0).split('.')[3]) 
    return lst    
            
# Merge two synonyms 
def merging_synsets(syns,w1,w2,sns1,sns2,max_score,loc,opt3):
    syns[loc].insert(0,max_score)
    syns[loc].insert(1,w1)
    syns[loc].insert(2,sns1.name())
    syns[loc].insert(3,w2)
    syns[loc].insert(4,sns2.name()) 
    if opt3=='d':
        lst=[]
        lst=extract_derivational_forms(sns1,w1,lst)
        lst=extract_derivational_forms(sns2,w2,lst) 
        #syns[loc]=list(sorted(set(syns[loc]),key=syns[loc].index))
        for e in lst:
            syns[loc].append(e) 
    loc=loc+1
    return syns,loc
    
# Finding most similar concept between two words 
def word_similarity(w1,w2,syns,loc,thr_sim,opt1,opt3):
    syn1=wn.synsets(w1, wn.NOUN or wn.ADJ)    
    syn2=wn.synsets(w2, wn.NOUN or wn.ADJ)    

    if len(syn1)>0 and len(syn2)>0:
        score=0; max_score=0; count=0
        sns1=syn1[0]
        sns2=syn2[0] 
        for i in range (0,len(syn1)):
            for j in range (0,len(syn2)):
                if opt1=='1':                                      # Wu-Palmer Similarity. It can not be '0'. It ranges in (0,1] 
                    score=wn.wup_similarity(syn1[i],syn2[j])  
                elif opt1=='2':                                    # Path Similarity 
                    score=wn.path_similarity(syn1[i],syn2[j])
                elif opt1=='3':                                    # Leacock-Chodorow Similarity 
                    score=wn.lch_similarity(syn1[i],syn2[j],simulate_root=False)
                elif opt1=='4':                                    # Resnik Similarity
                    score=wn.res_similarity(syn1[i],syn2[j])    
                elif opt1=='5':                                    # Jiang-Conrath Similarity
                    score=wn.jcn_similarity(syn1[i],syn2[j])
                elif opt1=='6':                                    # Lin Similarity
                    score=wn.lin_similarity(syn1[i],syn2[j])

                if score>max_score:                         # Finding the maximum score              
                    max_score=score
                    sns1=syn1[i] 
                    sns2=syn2[j] 
                    if max_score>=thr_sim:                     # Storing all the synset pairs that have scores > threshold 
                        syns,loc=merging_synsets(syns,w1,w2,sns1,sns2,max_score,loc,opt3)                       
                        count=count+1 
        if count==0:                           # Storing the synset that has maximum score but the score < threshold
            syns,loc=merging_synsets(syns,w1,w2,sns1,sns2,max_score,loc,opt3)   
    return syns,loc 

# Clustering of term sets
def clustering_terms(processed_terms,loc,thr_merge,flag): 
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
        return clustering_terms(processed_terms,loc,thr_merge,0)    # Iteration 
    else:                   
        return processed_terms

# Remove duplicates and find frequency of all unique terms in a cluster 
def process_cluster(data,all_terms,unique_terms,lst,rank):
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
        print sent
        if unique_terms.count(elm)>0:
            unique_terms[unique_terms.index(elm)]='' 
            
    new_lst.append(sent)                                       # Print the sentences where the terms in a cluster appear
    new_lst.append(rank)                                       # Rank of the cluster according to wordnet similarity 
    new_lst.append(cn)
    for elm in lst:
        new_lst.append(elm)  
    return unique_terms,new_lst  

# Loading the clusters into files aftre mapping each word to its actual form 
def store_cluster(data,path,clusters,loc,all_terms,unique_terms,stems,all_processed_terms,opt2,opt3):
    flr = open(path, 'wb') 
    wr = csv.writer(flr, delimiter=',',dialect='excel')  
    word_clusters=[] 
    count=0; rank=1
    for rw in clusters[0:loc]: 
        if rw[0]!='':
            lst=[];
            if opt3=='s':                           # Mapping the stems to the roots     
                for i in range(0,len(rw)):                
                    for j in range (0,len(stems)): 
                        if rw[i]==stems[j]:
                            lst.append(all_processed_terms[j]) 
            else:                                  # If stemming is not used 
                for i in range(0,len(rw)):
                    lst.append(rw[i])
            count=count+1 
            unique_terms,new_lst=process_cluster(data,all_terms,unique_terms,lst,rank) 
            rank=rank+1
            word_clusters.append(new_lst)
#            wr.writerow(new_lst)                    
    for elm in unique_terms:                        # Finding the frquencies of the single terms 
        if elm!='':
            lst=[] 
            lst.append(elm) 
            if opt3=='d': 
                if opt2=='b':
                    syn=wn.synsets(elm, wn.NOUN or wn.ADJ)
                elif opt2=='a':
                    syn=wn.synsets(elm, wn.ADJ)
                elif opt2=='n':
                    syn=wn.synsets(elm, wn.NOUN)    
                if len(syn)>0:
                    for i in range(0,len(syn)):
                        lst=extract_derivational_forms(syn[i],elm,lst)  
            unique_terms,new_lst=process_cluster(data,all_terms,unique_terms,lst,rank) 
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
def main():
#    text=open('what_qualities.txt', 'rb')
#    path1='what_qualities_synsets.csv'
#    path2='what_qualities_synsets_clusters.csv'

#    text=open('why.txt', 'rb')
#    path1='why_synsets.csv'
#    path2='why_synsets_clusters.csv'

#    text=open('what_do_you_think_more_dengerous.txt', 'rb')
#    path1='what_do_you_think_more_dengerous_synsets.csv'
#    path2='what_do_you_think_more_dengerous_synsets_clusters.csv'

    text=open('what_do_you_think_obese.txt', 'rb')
    path1='what_do_you_think_obese_synsets.csv'
    path2='what_do_you_think_obese_synsets_clusters.csv'

#    text=open('what_have_your_experiences.txt', 'rb')
#    path1='what_have_your_experiences_synsets.csv'
#    path2='what_have_your_experiences_clusters.csv'
 
#    text=open('what_do_you_think_more_dengerous.txt', 'rb')
#    path1='what_do_you_think_more_dengerous_synsets.csv'
#    path2='what_do_you_think_more_dengerous_clusters.csv'
 
#    text=open('is_it_better_worse.txt', 'rb')
#    path1='is_it_better_worse_synsets.csv'
#    path2='is_it_better_worse_clusters.csv'
                      
    opt1 = raw_input("Choose a similarity measure to identify relation between synsets: \n\t '1' Wu-Palmer Similarity \n\t '2' Path Similarity \n\t '3' Leacock-Chodorow (LCH) Similarity \n\t '4' Resnik Similarity \n\t '5' Jiang-Conrath Similarity \n\t '6' Lin Similarity \n") 
    opt2 = raw_input("Choose to process further: \n\t 'a' for adjectives \n\t 'n' for nouns \n\t 'b' for both \n")
    opt3 = raw_input("Choose to process further: \n\t 's' for stemming \n\t 'd' for derivationally related forms\n")
    
    text=''.join(text)
    data=[]; nouns=[]; adjectives=[]; all_terms=[] 

    sentences = [p for p in text.split('\n') if p]
    for s in sentences:  
        data.append(s)
        for w,pos in pos_tag(word_tokenize(str(s))):
            all_terms.append(w.lower()) 
            if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                 nouns.append(w.lower())
            elif (pos == 'JJ' or pos == 'JJR' or pos == 'JJS'):
                 adjectives.append(w.lower())
                 
    data=[x.lower() for x in data] 
    thr_merge=50                            # Threshold on percentage of intersection between two clusters    
    if opt1=='1':
        thr_sim=0.9                         # WUP similarity threshold 
    elif opt1=='2':
        thr_sim=0.4                         # Path similarity threshold
    elif opt1=='3':
        thr_sim=2.8                         # Leacock-Chodorow (LCH) similarity threshold 

    if opt2=='a':                   # Process adjectives only
        terms=adjectives    
    elif opt2=='n':                 # Process nouns only
        terms=nouns
    elif opt2=='b':                 # Process both adjectives and nouns
        terms=adjectives+nouns

# Synsets generation   
    loc=0
    unique_terms=list(set(terms))
#    print unique_terms 
    ln=len(unique_terms)
    ln=ln*(ln-1)/2 
    syns = [[]for x in range(ln)]     
    for i in range(0,len(unique_terms)-1):
        for j in range(i+1,len(unique_terms)):
            syns,loc=word_similarity(unique_terms[i],unique_terms[j],syns,loc,thr_sim,opt1,opt3) # Finding most similar synset 
          
    syns=sorted(syns[0:loc], key = lambda x: x[0], reverse=True)   # Sort the synsets in decreasing order of scores
#    print syns[0] 
    heading = []
    heading.append('Score')
    heading.append('Term1')
    heading.append('Term2')
    heading.append('Synset1')
    heading.append('Synset2')        
    processed_terms = [[]for x in range(ln)]
    loc=0
    flr = open(path1, 'wb')
    wr = csv.writer(flr, delimiter=',',dialect='excel')     # Writing the synsets to a file along with the words 
    wr.writerow(heading)
#   Writing the words with their synsets
    stems=[]; all_processed_terms=[] 
    for rw in syns:
        wr.writerow(rw) 
        if rw[0]>=thr_sim:                      
            for i in range(1,len(rw)): 
                if opt3=='s':                                   # Perform stemming
                    stm=PorterStemmer().stem_word(rw[i])             
                    if all_processed_terms.count(rw[i])==0:
                        stems.append(stm) 
                        all_processed_terms.append(rw[i]) 
                    processed_terms[loc].append(str(stm))
                else:
                    processed_terms[loc].append(str(rw[i]))   # Consider actual words                    
            loc=loc+1
    flr.close()

    clusters=clustering_terms(processed_terms,loc,thr_merge,0)         # Clustering 
    count=store_cluster(data,path2,clusters,loc,all_terms,unique_terms,stems,all_processed_terms,opt2,opt3) 
    print 'No. of Clusters: '+ str(count)

if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf8')
    main()
