#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 17:17:40 2020

@author: basut
"""

from qualitative_text_analysis import qualitative_text_analysis


clf=qualitative_text_analysis('/Users/basut/myvoice/what_qualities.txt','w','b','d')
clf.get_summary()
