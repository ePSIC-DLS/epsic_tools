# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:37:27 2020

@author: gys37319
"""

import os
import glob


def visit_dir(beamline, year, visit):
    """
    Returns list of directories in visit
    
    Parameters
    ----------
    beamline: string (e.g. 'e01', 'e02')
    year: string 
    visit: string

    Returns
    -------
    dir_list: list of strings 
        list of subdirectories
    """
    directory = r'/dls/'+ beamline + r'/data/' + year +r'/' +visit 
    dir_list =  [x[0] for x in os.walk(directory)]
    number_list(dir_list)
    return dir_list
    
def list_files(dir_list, dir_num, search_str = '*.*'):
    """
    Returns list files in subdirectory matching search condition
    
    Parameters
    ----------
    dir_list: list of directory strings
    dir_num: int of dir from dir_list to search 
    search_str: string for search matching

    Returns
    -------
    file_list: list  
        list of files
    """
    #print(dir_list[dir_num])
    data_dir = dir_list[dir_num] + r'/'
    file_list = glob.glob(data_dir + search_str)
    number_list(file_list)
    return file_list
    
def number_list(a_list):
    """
    Prints numbered list
    
    Parameters
    ----------
    a_list: list

    Returns
    -------
    none
    """
    file_num = 0
    for this_file in a_list:
        print('[', file_num, '] : ', this_file)
        file_num += 1