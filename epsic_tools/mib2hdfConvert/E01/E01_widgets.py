# code stared 07/11/2025 t 13:51:27 by ejr78941

import re
import numpy as np
import time
import os
import glob
import json
from pathlib import Path
import datetime
from IPython.display import display, clear_output
import ipywidgets.widgets as w
import ipywidgets
import matplotlib.pyplot as plt
import h5py
import traceback

import subprocess



#other functions:

def delete_ptycho_flagging_files(basedir,year,session,subfolder,verbose=False):
    '''This function is for deleting the autoptycho_is_done.txt files which prevent recontructions
       from being run more than once, This code is not yet complete'''

    '''determine path path being searched and change directory to that path for glob'''
    dest_path = f'/{basedir}/{year}/{session}/raw/{subfolder}'
    os.chdir(dest_path)
    print('File being searched: ' + dest_path)


def _create_ptyrex_bash_submit(json_files, script_folder, node_type, ptycho_time, verbose=False):
    bash_ptyrex_path = []

    for num, x in enumerate(json_files):
        if verbose:
            print(f'\n{str(num)}: {x}\n')
        #time_stamp_index = x.find('/pty_out')
        #time_stamp = x[time_stamp_index - 6:time_stamp_index]
        time_stamp = num
        if verbose:
            print(f'\ntime_stamp = {time_stamp}\n')
        '''rememeber to self.script_save_path to save these files to the script folder'''
        bash_ptyrex_path.append(os.path.join(script_folder, f'{time_stamp}_ptyrex_submit.sh'))
        if verbose:
            print(f'{num}: {bash_ptyrex_path[num]}')
        if True:
            with open(bash_ptyrex_path[num], 'w') as f:
                f.write('#!/usr/bin/env bash\n')
                f.write('#SBATCH --partition=cs05r\n')
                f.write('#SBATCH --job-name=ptyrex_recon\n')
                f.write('#SBATCH --nodes 1\n')
                f.write('#SBATCH --tasks-per-node=4\n')
                f.write('#SBATCH --cpus-per-task 1\n')
                f.write('#SBATCH --gpus-per-node=4\n')
                f.write(f'#SBATCH --time {ptycho_time}\n')
                f.write('#SBATCH --mem 0G\n\n')
                f.write(f'#SBATCH --constraint=NVIDIA_{node_type}\n')

                f.write(f"#SBATCH --error={script_folder}{os.sep}%j_error.err\n")
                f.write(f"#SBATCH --output={script_folder}{os.sep}%j_output.out\n")

                f.write(f"cd /home/ejr78941/ptyrex_temp_5/PtyREX")

                f.write('\n\nmodule load python/cuda11.7\n\n')
                f.write('module load hdf5-plugin/1.12\n\n')

                f.write(f"mpirun -np 4 ptyrex_recon -c {json_files[num]}\n")
    return bash_ptyrex_path

def _create_dm4_bash(dm4_files, binning, script_folder, verbose=False):
    bash_path = []

    for num, x in enumerate(dm4_files):
        if verbose:
            print(f'\n{str(num)}: {x}\n')

        time_stamp = num
        if verbose:
            print(f'\ntime_stamp = {time_stamp}\n')
        '''rememeber to self.script_save_path to save these files to the script folder'''
        bash_path.append(os.path.join(script_folder, f'{time_stamp}_dm4_convert_submit.sh'))
        if verbose:
            print(f'{num}: {bash_path[num]}')
        if True:
            with open(bash_path[num], 'w') as f:
                f.write('#!/usr/bin/env bash\n')
                f.write('#SBATCH --partition=cs05r\n')
                f.write('#SBATCH --job-name=data_reformat\n')
                f.write('#SBATCH --nodes 1\n')
                f.write('#SBATCH --tasks-per-node=1\n')
                f.write('#SBATCH --cpus-per-task 1\n')
                f.write(f'#SBATCH --time 24:00:00\n')
                f.write('#SBATCH --mem 0G\n\n')
                f.write(f"#SBATCH --error={script_folder}{os.sep}%j_error.err\n")
                f.write(f"#SBATCH --output={script_folder}{os.sep}%j_output.out\n")
                f.write('\n\nmodule load python/epsic3.10\n\n')

                f.write(f"python /dls_sw/e02/software/epsic_tools/epsic_tools/mib2hdfConvert/E01/Convert_data_inital.py {dm4_files[num]} {binning}\n")
    return bash_path

def _create_flagging_text_files(submit_ptyrex_job, tmp_list, verbose=False):
    '''
    :param submit_ptyrex_job: a test whether ptyrex jobs should be submitted, used here as addiotnional test
    to check whether this function should run or not
    :param tmp_list: a list of json files, this used to determine the directories where should auto ptycho flagging
    should be generated
    :param verbose: this is used to determine whether full or minial printout should be used
    :return:
    '''
    if submit_ptyrex_job:
        # assuming the paths given are the json_paths
        test_string = 'autoptycho_is_done.txt'
        default_string = 'this data set has been auto reconstructed already and therefore will skipped in all proceeding auto recons, please detele to restore auto functionality'
        for x in tmp_list:
            a = x.find('pty_out/') + len('pty_out/')
            flagging_files = (x[:a] + test_string)
            if verbose:
                print(f'\nflag file name: {flagging_files}\n')
            with open(flagging_files, 'w') as f:
                f.write(default_string)
            f.close()

def _ptyrex_ssh_submit(bash_list, submit_ptyrex_job, tmp_list=None, verbose=False):
    '''
    :param bash_list:
    :param submit_ptyrex_job:
    :param tmp_list:
    :param verbose:
    :return:
    '''
    if submit_ptyrex_job:
        sshProcess = subprocess.Popen(['ssh',
                                       '-tt',
                                       'wilson'],
                                      stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,
                                      universal_newlines=True,
                                      bufsize=0)
        sshProcess.stdin.write("ls .\n")
        sshProcess.stdin.write("echo END\n")
        for num, x in enumerate(bash_list):
            sshProcess.stdin.write(f"sbatch {x}\n")
        sshProcess.stdin.write("uptime\n")
        sshProcess.stdin.write("logout\n")
        sshProcess.stdin.close()

        for line in sshProcess.stdout:
            if line == "END\n":
                break
            print(line, end="")

        # to catch the lines up to logout
        for line in sshProcess.stdout:
            print(line, end="")
    try:
        if tmp_list != None:
            _create_flagging_text_files(submit_ptyrex_job, tmp_list, verbose)
    except:
        print(traceback.format_exc())

class E01_auto_process():
    def __init__(self,E01_convert=False,json_files=False, run_ptyrex=False):

        if E01_convert:
            self._convert_dm_2_hdf5()

        elif json_files:
            self.widget_Gen_json_files()

        elif run_ptyrex:
            self._ptyrex_submit()

        else:
            print('select one of the following modes:\n1: E01_convert=True\njson_files=True')


    def prefill_boxes(self):
        '''
        This function is used to prefill the basedir, year and session boxes in 
        the ipython widgets such that users do not have repeatly enter in the 
        same information if they are running different cells within the same notebook
        this only works if the folder which the note book is in is associated with a 
        particular user session and this can work with staged data as well. so is based
        strongly on os.chdir function.
        '''
        #ToDo make these varibles be stored in self such that once it has been filled out one

        st = {"description_width": "initial"}
        current_dir = os.getcwd()
        if '/'.join(current_dir.split('/')[1:4]) == 'dls/e01/data':
            basedir = w.Text(value='/'.join(current_dir.split('/')[1:4]), description='Base data directory path:', style=st)
            year = w.Text(value=current_dir.split('/')[4],description='Year:', style=st)
            session = w.Text(value=current_dir.split('/')[5],description='Session:', style=st)
        elif current_dir == 'dls/staging/dls/e01/data':
            basedir = w.Text(value='/'.join(current_dir.split('/')[1:5]), description='Base data directory path:', style=st)
            year = w.Text(value=current_dir.split('/')[6],description='Year:', style=st)
            session = w.Text(value=current_dir.split('/')[7],description='Session:', style=st)
        else:
            basedir = w.Text(value="/dls/e01/data", description='Base data directory path:', style=st)
            year = w.Text(description='Year:', style=st)
            session = w.Text(description='Session:', style=st)
        return basedir, year, session

            
    def find_files(self,basedir,year,session,subfolder,output, Choose_binning, convert_dm, verbose):
        #predefing list and varibles to be used in this section of the code
        flagging_text = 'conversion_is_done.txt'
        self.already_converted_list = []
        self.diffraction_dm_list = []
        self.to_convert = []
        data_tag = []
        converted_tag = []
        count = 0 #count is required as sometimes there are multiple files with Diffraction SI.dm4 at the end
        num = -1
        '''
        using the inputs provided construct the path to the data that the user
        is instered in
        '''
        tmpor = basedir + '/' + year + '/' + session
        if basedir == '' or year == '' or session == '' or subfolder == '':
            print(f'***\ncurrent path: /{basedir}/{year}/{session}/raw/{subfolder}/\n***')
        else:
            print(f'current path: /{basedir}/{year}/{session}/raw/{subfolder}/')
            self.output.value = f'/{basedir}/{year}/{session}/raw/{subfolder}/'

            '''
            now that path has been constructed use it as starting point
            for glob to find all of saved ptychography data and the dat
            which has already been converted into hdf5
            '''
            
            try:
                os.chdir(self.output.value)
                self.script_path = self.output.value + 'scripts/'
                if verbose:
                    print('script path: %s' % self.script_path)
                if not os.path.exists(self.script_path):                    
                    os.makedirs(self.script_path)
                if verbose:
                    print(f'\nsearching for dm4 files...')
                for num, file in enumerate(sorted(list(glob.glob('*/**Diffraction SI.dm4*')))):
                    if file.split('/')[1] == 'Diffraction SI.dm4':
                        self.diffraction_dm_list.append(self.output.value+file)
                        data_tag.append(self.diffraction_dm_list[count].split('/')[-2])
                        if verbose:
                            print(f'tag {num}: {data_tag[count]}')
                            print(f'file {num}: {self.diffraction_dm_list[count]}')
                        '''increment count'''
                        count = count + 1
                disp_num = num + 1
                if verbose:
                    print(f'\nSearching for converted files (data.hdf5)...')
                for num, file in enumerate(sorted(list(glob.glob('*/**data.hdf5')))):
                    self.already_converted_list.append(self.output.value+file)
                    converted_tag.append(self.already_converted_list[num].split('/')[-2])
                    if verbose:
                        print(f'tag {num}: {converted_tag[num]}')
                        print(f'converted file {num}: {self.already_converted_list[num]}')
                '''determine the data which has already been converted using the difference method within python sets to
                   cross reference the tags with each other (typically tags are something like SI-002)'''
                disp_num = disp_num - (num + 1)
                data_tag_set = set(data_tag)
                to_convert_tag = list(data_tag_set.difference(converted_tag))
                to_convert_tag.sort()
                if verbose:
                    print('\nThe following files have not been converted: ')
                for num, x in enumerate(to_convert_tag):
                    self.to_convert.append(self.diffraction_dm_list[data_tag.index(x)])
                    if verbose:
                        print(f'{num}: {self.diffraction_dm_list[data_tag.index(x)]}')
                 
                
                print('***\nThe number of unconverted files in this directory is: %i\n***' % disp_num)
                print(f'\nbinning value chosen: {eval(Choose_binning)}')
                if convert_dm:
                    if verbose:
                        print(f'\n***\nConverting selected data from dm4 to hdf5\n***')
                    try:
                        bash_list = _create_dm4_bash(self.to_convert, Choose_binning, self.script_path, verbose)
                        _ptyrex_ssh_submit(bash_list, convert_dm, verbose=verbose)
                    except:
                        print(traceback.format_exc())
                    #self.convert_dm.value=False
  
            except Exception:
                print(traceback.format_exc())
                print(f'\n\nthe inputted path does not match a real location')



        
    def _convert_dm_2_hdf5(self):
        st = {"description_width": "initial                     "}
        self.basedir, self.year, self.session = self.prefill_boxes()
        #self.basedir = w.Text(description='Base data directory path:', style=st,value='dls/e01/data')
        #self.year = w.Text(description='Year:', style=st,value='2024')
        #self.session = w.Text(description='Session:', style=st,value='cm37230-5')
        self.subfolder = w.Text(description='subfolder:', style=st,disabled=False,value='')
        self.output = w.Text(description='output:', style=st,disabled=True,value='Empty')
        self.Choose_binning = w.Dropdown(options=['1','2','4','8'], value='2', description='choose a binning value', style=st)
        self.convert_dm = w.Checkbox(value=False, description='Convert data to hdf5', style=st)
        self.verbose = w.Checkbox(value=False, description='Check this for debugging and error printing', style=st)

        

        layout = w.Layout(
            display='grid',
            grid_template_columns = '50% 50%',
            grid_gap='10px 10px')
        
        boxes = [self.basedir, self.year, self.session, self.subfolder, self.output, self.Choose_binning, self.convert_dm, self.verbose]

        grid = w.GridBox(children = boxes, layout = layout)
        
        
        
        self._test = ipywidgets.interactive_output(self.find_files,
                                {'basedir': grid.children[0],
                                'year': grid.children[1],
                                'session': grid.children[2],
                                'subfolder': grid.children[3],
                                'output': grid.children[4],
                                'Choose_binning': grid.children[5],
                                'convert_dm': grid.children[6],
                                'verbose': grid.children[7]})
        display(grid,self._test)



    def create_E01_json(self,hdf_data_list,template_json='',verbose=True):
         num = -1
         if template_json == '':
             template_json = '/dls_sw/e02/software/epsic_tools/epsic_tools/mib2hdfConvert/E01/E01_template.json'
             if verbose:
                print(f'\nusing the standard json as no template was entered, see the following location:\n {template_json}\n')
         else:
             if verbose:
                 print(f'using custom json, using the json at the following location:\n{template_json}')
         self.json_dest = []
         self.json_name = []
         for num, file in enumerate(hdf_data_list):
             '''define output path'''
             self.json_dest.append(file.replace(file.split('/')[-1],'pty_out/'))
             if verbose:
                 print(f'{num}. json destination: {self.json_dest[num]}')
             '''obtain meta data from the hdf5 file'''
             if verbose:
                 print('\n***\nopenning hdf5 file\n***\n')
             with h5py.File(file,'r') as f:
                 scan_shape = [f['data']['frames'].shape[0],f['data']['frames'].shape[1]]
                 cropping   = [f['data']['frames'].shape[2],f['data']['frames'].shape[3]]
                 # ToDo fix the below as I am being very lazy I should take directly from the input the x and y pixel but here I assuming their equal
                 # which they should be...
                 pixel_size = [np.round(f['/detector/pixel_size'],9),np.round(f['/detector/pixel_size'],9)]
                 try:
                     step_size = [f['/scan/scale'][()]*1e-9,f['/scan/scale'][()]*1e-9]
                 except:
                     step_size = 4e-11
                     print(f'scan step size was not saved to the hdf5 file, assumming a step size of {step_size}')
             '''duplicate the template json and replace the required values'''
             if verbose:
                 print('\nattempting to generate json file...')
             with open(template_json,'r+') as json_file:
                pty_expt = json.load(json_file)
                pty_expt['base_dir'] = self.json_dest
                pty_expt['experiment']['data']['data_path'] = file
                pty_expt['process']['save_dir'] = self.json_dest[num]
                pty_expt['process']['save_prefix'] = 'E01_ptyrex_recon' + str(num)
                pty_expt['process']['PIE']['iterations'] = 25
                pty_expt['process']['PIE']['probe']['alpha'] = 0.1
                pty_expt['process']['PIE']['object']['alpha'] = 0.1
                pty_expt['process']['common']['scan']['N'] = scan_shape
                pty_expt['process']['common']['scan']['dR'] = step_size
                pty_expt['process']['common']['detector']['crop'] = cropping
                pty_expt['process']['common']['detector']['pix_pitch'] = pixel_size
             
             '''determine the save name and then save the json file'''
             if verbose:
                 print(f'{num}: saving the json file to the following location: {self.json_dest[num]}')
             if not os.path.exists(self.json_dest[num]):
                 os.makedirs(self.json_dest[num])
                 if verbose:
                     print(f'creating the following folder: {self.json_dest[num]}')
             self.json_name.append(self.json_dest[num] + f'E01_auto.json')
             if verbose:
                 print(f'{num}. json file name: {self.json_name[num]}')
             with open(self.json_name[num],'w') as json_write:
                json.dump(pty_expt,json_write,indent=4)
         print(f'\n***\ncreated {num+1} json files\n***')



    def gen_json_files(self,basedir, year, session, subfolder, output, data_name, generate_jsons, verbose):
        '''
        function to take widget inputs to determine a path to converted hdf5 files and then
        create corresponding json file which then allow for ptyrex reconstruction
        '''
        self.hdf_data_list = []
        '''
        using the inputs provided construct the path to the data that the user
        is instered in
        '''
        tmpor = basedir + '/' + year + '/' + session
        if basedir == '' or year == '' or session == '' or subfolder == '':
            print(f'current path: /{basedir}/{year}/{session}/raw/{subfolder}/')
        else:
            print(f'current path: /{basedir}/{year}/{session}/raw/{subfolder}/')
            self.output.value = f'/{basedir}/{year}/{session}/raw/{subfolder}/'

            '''
            now that path has been constructed use it as starting point
            for glob to find all of saved ptychography data
            '''
            try:
                os.chdir(self.output.value)
                if verbose:
                    print('***\nthe following data sets where found in this search:\n***\n')
                for num, file in enumerate(sorted(list(glob.glob(f'*/**{data_name}.hdf5*')))):
                        self.hdf_data_list.append(self.output.value + file)
                        if verbose:
                            print(f'{num}: {self.hdf_data_list[num]}')
                        
                if generate_jsons:
                    try:
                        self.create_E01_json(self.hdf_data_list,verbose=verbose)
                    except Exception:
                        print(traceback.format_exc())
            except:
                print(f'the inputted path does not match a real location')



    def widget_Gen_json_files(self):
        st = {"description_width": "initial                     "}
        self.basedir, self.year, self.session = self.prefill_boxes()
        #self.basedir = w.Text(description='Base data directory path:', style=st,value='dls/e01/data')
        #self.year = w.Text(description='Year:', style=st,value='2024')
        #self.session = w.Text(description='Session:', style=st,value='cm37230-5')
        self.subfolder = w.Text(description='subfolder:', style=st,disabled=False,value='')
        self.output = w.Text(description='output:', style=st,disabled=True,value='Empty')
        self.data_name = w.Text(description='name of hdf5 file to use:', style=st,disabled=False,value='')
        #self.json_name = w.Text(description='name of the outputted json files:', style=st,disabled=False,value='')
        self.generate_jsons = w.Checkbox(value=False, description='generate json files', style=st)
        self.verbose = w.Checkbox(value=False, description='Check this for debugging and error printing', style=st)

        

        layout = w.Layout(
            display='grid',
            grid_template_columns = '50% 50%',
            grid_gap='10px 10px')
        
        boxes = [self.basedir, self.year, self.session, self.subfolder, self.output, self.data_name, self.generate_jsons, self.verbose]

        grid = w.GridBox(children = boxes, layout = layout)
        
        
        
        self._json = ipywidgets.interactive_output(self.gen_json_files,
                                {'basedir': grid.children[0],
                                'year': grid.children[1],
                                'session': grid.children[2],
                                'subfolder': grid.children[3],
                                'output': grid.children[4],
                                'data_name': grid.children[5],
                                'generate_jsons': grid.children[6],
                                'verbose': grid.children[7]})
        display(grid,self._json)



        


    def _ptyrex_paths(self,basedir, year, session, subfolder, ptycho_config_name, create_ptycho_bash_script_check,
                      node_type, ptycho_time, submit_ptyrex_job, delete_flaggin_files, verbose=False):
        '''define the source path'''
        src_path = f'/{basedir}/{year}/{session}/raw/{subfolder}'
        script_folder = f'/{basedir}/{year}/{session}/raw/{subfolder}/scripts'

        tmp_list = []
        test_string = 'autoptycho_is_done.txt'

        '''use glob and os to find the meta data files'''
        if basedir == '' or year == '' or session == '' or subfolder == '' or ptycho_config_name == '':
            print('\nwaiting for the folowing inputs: basedir, year, session, subfolder and config_name\n')
            print(f'current path: /{basedir}/{year}/{session}/raw/{subfolder}')
        else:
            '''try statement is used here to catch the case where all parameters have inputs but are still not correct mid type'''
            try:
                os.chdir(src_path)
                print(f'current path: /{basedir}/{year}/{session}/raw/{subfolder}\n')
                if verbose:
                    print(f'json files found in the current path are listed below:\n')
                for num, file in enumerate(sorted(list(glob.glob('*/*/*' + ptycho_config_name + '.json')))):
                    '''debug statement to check file paths'''
                    if verbose == True:
                        print(str(num) + ': ' + os.path.join(src_path, file))
                        # print(str(num) + ': ' + os.path.join(src_path,file.replace(ptycho_config_name + '.json', test_string)))

                    '''check whether the data has been processed before - if it has then it should have autoptycho_is_done text file'''
                    #ToDo fix this to test whether even in the case where only a subsection of the code matches (it must be a whole match)
                    #for example ptycho_Sample and 2_ptycho_Sample will be brought up by ptycho_Sample as they share strings
                    if os.path.exists(os.path.join(src_path, file.replace(ptycho_config_name + '.json', test_string))):
                        if verbose:
                            print(
                                '\nskipping json as it already been processed, to process this again please delete flagging files\n')
                    else:
                        tmp_list.append(os.path.join(src_path, file))

                '''check whether the scripting folder exists, if not create the folder'''
                if os.path.exists(script_folder) == False:
                    os.mkdir(script_folder)

                '''delete auto ptycho flagging  files if there are any'''
                if delete_flaggin_files:
                    delete_ptycho_flagging_files(basedir,year,session,subfolder,verbose)
                
                '''Create bash scripts for each of the json files'''
                if create_ptycho_bash_script_check:
                    print('\n***\nFound json files and creating bash scripts...\n***\n')
                    bash_list = _create_ptyrex_bash_submit(tmp_list, script_folder, node_type, ptycho_time, verbose)
                '''submit the bash scripts to the cluster'''
                if submit_ptyrex_job:
                    _ptyrex_ssh_submit(bash_list, submit_ptyrex_job, tmp_list)
                    if verbose:
                        print(f'submited jobs to wilson')
                if create_ptycho_bash_script_check and submit_ptyrex_job:
                    return tmp_list, bash_list
            
            except:
                print(f'current path: /{basedir}/{year}/{session}/processing/Merlin/{subfolder}')
                print(f'\nEither the entered path does not exist, Path searched: {src_path}, or a different error has occurred\n')



    def _ptyrex_submit(self):
        st = {"description_width": "initial"}
        self.basedir, self.year, self.session = self.prefill_boxes()
        #self.basedir = w.Text(description='Base data directory path:', style=st,value='dls/e01/data')
        #self.year = w.Text(description='Year:', style=st,value='2024')
        #self.session = w.Text(description='Session:', style=st,value='cm37230-5')
        self.subfolder = w.Text(description='Subfolder:', style=st,disabled=False,value='')
        self.ptycho_config_name = w.Text(value='E01_auto',description='Name of the json file to process:', style=st)
        self.create_ptycho_bash_script_check = w.Checkbox(value=False, description='Create PtyREX bash script', style=st)
        self.node_type = w.Dropdown(options=['Volta', 'Pascal'], value='Pascal', description='type of gpu to use')
        self.ptycho_time = w.Dropdown(options=['00:15:00','00:30:00', '01:00:00', '02:00:00', '04:00:00', '08:00:00'],value='00:30:00',description='Processing time: (HH:MM:SS)')
        self.submit_ptyrex_job = w.Checkbox(value=False, description='Submit PtyREX job', style=st)
        self.delete_flaggin_files = w.ToggleButton(value=False,description='delete flagging files', disabled=False, button_style='')
        self.verbose = w.Checkbox(value=False, description='Check this for debugging and error printing', style=st)




        layout = w.Layout(
            display='grid',
            grid_template_columns = '50% 50%',
            grid_gap='10px 10px')
        
        boxes = [self.basedir, self.year, self.session, self.subfolder, self.ptycho_config_name, 
                 self.create_ptycho_bash_script_check, self.node_type, self.ptycho_time, 
                 self.submit_ptyrex_job, self.delete_flaggin_files, self.verbose]

        grid = w.GridBox(children = boxes, layout = layout)

        self.ptyrex_paths = ipywidgets.interactive_output(self._ptyrex_paths,
                                          {'basedir': grid.children[0],
                                          'year': grid.children[1], 
                                          'session': grid.children[2],
                                          'subfolder': grid.children[3],
                                          'ptycho_config_name': grid.children[4],
                                          'create_ptycho_bash_script_check': grid.children[5],
                                          'node_type': grid.children[6],
                                          'ptycho_time': grid.children[7],
                                          'submit_ptyrex_job': grid.children[8],
                                          'delete_flaggin_files': grid.children[9],
                                          'verbose': grid.children[10]})
        display(grid,self.ptyrex_paths)
        
