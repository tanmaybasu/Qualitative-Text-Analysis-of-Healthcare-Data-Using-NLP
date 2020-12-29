#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday October 3, 2016 @ 15:47:21  

@author: Tanmay Basu
"""

from qualitative_text_analysis import qualitative_text_analysis


clf=qualitative_text_analysis(path='/Users/basut/myvoice/what_qualities.txt',wordnet_metric='w',pos='b',transformation='d')
clf.get_summary()

