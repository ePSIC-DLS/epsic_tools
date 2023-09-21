#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import fnmatch
from pprint import pprint
import boto3
import configparser


class ePSIC_S3(object):
    ''' 
    object to interact with S3 file system
    '''
    
    def __init__(self, beamline, year, proposal):
    
        # Parsing Config file
        config = configparser.ConfigParser()
        config.read('/dls/science/groups/e02/IRIS_S3/rclone.conf' )
         
        # Creating a session with keys
        session = boto3.session.Session(
                    aws_access_key_id = config['epsic_s3']['access_key_id'],
                    aws_secret_access_key = config['epsic_s3']['secret_access_key'])
        # Connecting to session, with the endpoint defined in the config
        self.s3 = session.resource('s3',endpoint_url = config['epsic_s3']['endpoint'])
        #bucket_address = beamline + '_' + year + '_' + proposal # can we wildcard here?
        bucket_list = self.s3.buckets.all()
        bucket_names = [bucket.name for bucket in bucket_list]
        shift = 0
        
        if len(proposal) == 7:
            bucket_address = beamline + '_' + year + '_' + proposal
            matching = [s for s in bucket_names if bucket_address in s]
            if not matching:
                print("No session matching ", beamline, year, proposal) 
            else:
                if len(matching) == 1:
                    bucket_address = matching[0]
                    print('Session address:',  bucket_address)
                else:
                    session_numbers = [int(this_session[-1]) for this_session in matching]
                    session_numbers_str = [(this_session[-1]) for this_session in matching]
                    while not int(shift) in session_numbers:
                        str_session_numbers = (',').join(session_numbers_str)
                        shift = input("For proposal " + proposal + ", enter your session number: " + str_session_numbers + ": ")
                        bucket_address = bucket_address + '-' + str(shift)
                        print('Session address:',  bucket_address) # should we check here that it exists?
        else:
            print('please enter full session number')
        self.bucket = self.s3.Bucket(bucket_address)
        
        
    def data_types(self):
        objects = self.bucket.objects.all()
        key_list = [this_obj.key.split('/')[0] for this_obj in objects]
        key_list = list(set(key_list))
        pprint(key_list)
    
    def samples(self, data_set):
        data_keys = []
        prefix_str = data_set + '/'
        for obj in self.bucket.objects.filter(Prefix=prefix_str):
            data_keys.append(obj.key)
        data_list = []
        for this_key in data_keys:
            data_list.append(this_key.split('/')[1] +'/' +  this_key.split('/')[2])
        data_list = list(set(data_list))
        sample_list = [this_data.split('/')[0] for this_data in data_list]
        sample_list = list(set(sample_list))
        pprint(sample_list)
        
    def get_data_list(self, key_search='None'):
        data_list = []
        for obj in self.bucket.objects.all():
            key = obj.key
            if key_search == 'None':
                data_list.append(key)
            else:
                if key_search in key:
                    data_list.append(key)
        return data_list
            
            
            
def print_buckets():
    #this function will be killed 
        # Parsing Config file
    config = configparser.ConfigParser()
    config.read('/dls/science/groups/e02/IRIS_S3/rclone.conf' )
     
    # Creating a session with keys
    session = boto3.session.Session(
                aws_access_key_id = config['epsic_s3']['access_key_id'],
                aws_secret_access_key = config['epsic_s3']['secret_access_key'])
    # Connecting to session, with the endpoint defined in the config
    s3 = session.resource('s3',endpoint_url = config['epsic_s3']['endpoint'])
    #bucket_address = beamline + '_' + year + '_' + proposal # can we wildcard here?
    bucket_list = s3.buckets.all()
    bucket_names = [bucket.name for bucket in bucket_list]
    shift = 0
    pprint(bucket_names)
            
