import pathlib
import numpy as np
import hyperspy.api as hs
from scipy import interpolate
from matplotlib import widgets
import matplotlib.pyplot as plt
import hyperspy.api as hs
import ncempy.io




class Path(type(pathlib.Path())):
    def ls(self):
        return list(self.iterdir())
    def walk(self, inc_condition = None, exc_condition = None, max_depth=None):
        '''
        Find all the files contained within this folder recursively

        inc_condition (str): only include results containing this string
        exc_condition (str): don't include any results containing this string 
        max_depth (int): maximum depth of recursion, default None has no limit'''
        cont = True
        all_files = self.ls()
        depth = 1
        while cont==True:
            some_files = []
            for f in all_files:
                if f.is_dir():
                    [some_files.append(x) for x in f.ls()]
                else:
                    some_files.append(f)
            if depth == max_depth:
                cont = False
            else:
                cont = not np.all(np.asarray([x.is_file() for x in some_files]) == True)
                depth +=1
            all_files = some_files.copy()
        if inc_condition != None:
            all_files = [x for x in all_files if str(x).find(inc_condition) != -1]
        if exc_condition != None:
            all_files = [x for x in all_files if str(x).find(exc_condition) == -1]
        return all_files
    def redirect(self, end, index = 1):
        '''
        Replace the end target of the Path spliting at the i-th / from the end

        end (str): new target to append to path
        index (int): position to append new target

        eg. Path('a/b/c/d/e').redirect('f/g', 2) --> Path('a/b/c/f/g') '''
        if index != 0:
            return Path('/'.join(str(self).split('/')[:-index]) + f'/{end}')
        else:
            return Path(str(self)+f'/{end}')
    def mk(self, recursive = False):
        '''
        Make a directory for the path if one does not exist
        '''
        if recursive == False:
            if not self.exists():
                self.mkdir()
        if recursive == True:
            tpath = self
            count = -1
            while tpath.exists() == False:
                tpath = self.redirect('/', count+1)
                count += 1
            while count >= 0:
                p = self.redirect('/', count-1)
                if not p.exists():
                    p.mkdir()
                count -= 1


def flatten_nav(sig):
    shape = [sig.shape[0]*sig.shape[1]]
    for i in sig.shape[2:]:
        shape.append(i)
    return sig.reshape(shape)

def best_model_from_list(path_list):
    '''
    Takes a list of paths containing trained models and returns the path of the model with the lowest loss'''
    if len(path_list)==1:
        return path_list[0]
    else:
        best_loss = float(path_list[0].parts[-1].split('-')[-1].split('.hdf5')[0])
        best_ind = 0
        for i, p in enumerate(path_list[1:]):
            loss = float(p.parts[-1].split('-')[-1].split('.hdf5')[0])
            if loss < best_loss:
                best_loss = loss
                best_ind = i
        return path_list[best_ind]

def make_uniform(sig, non_uni_vals, n_uni_points):
    tot_x, tot_y = sig.data.shape[0], sig.data.shape[1]
    new_arr = np.zeros((tot_x, tot_y, n_uni_points))
    n_interp_x = np.linspace(non_uni_vals[0], non_uni_vals[-1], n_uni_points)
    for x_pos in range(tot_x):
        for y_pos in range(tot_y):
            t_interp_y = sig.data[x_pos,y_pos]
            interp_func = interpolate.splrep(non_uni_vals,t_interp_y, s=0)
            new_arr[x_pos, y_pos] = interpolate.splev(n_interp_x, interp_func)
    new_sig = hs.signals.Signal1D(new_arr)
    offset = n_interp_x[0]
    new_sig.axes_manager[2].offset= n_interp_x[0]
    new_sig.axes_manager[2].scale = (n_interp_x[-1] - offset)/n_uni_points
    return new_sig, n_interp_x

def resample_data(x, y, new_x_range, sampling_resolution =100, return_x = False):
    
    '''
    x: original x data (np array)
    y: original y data at each x point (np array)
    new_x_range: new lower x bound and new higher x bound (2 value tuple)
    sampling resolution: number of points to contain in a step of value 1 along the x axis (float)
    return_x: whether to return the new x range values (bool)
    
    returns:
    
    either - new_y
    or - (new_x, new_y)
    
    '''

    xl, xu = new_x_range
    new_x_vals = np.linspace(xl, xu, (xu-xl)*sampling_resolution)
    
    if return_x == False:

        return interpolate.interp1d(x, y)(new_x_vals)
    
    if return_x == True:

        return new_x_vals, interpolate.interp1d(x, y)(new_x_vals)
class RectangleSelector(widgets.RectangleSelector):
    def get_lims(self, scale = 1):
        extent = self.extents
        return np.asarray(extent)*scale


def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    print(" The button you used were: %s %s" % (eclick.button, erelease.button))


def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


        
def draw_rectangle_selector(im):
    fig, current_ax = plt.subplots()                 # make a new plotting range
    plt.imshow(im)
    print("\n      click  -->  release")

    # drawtype is 'box' or 'line' or 'none'
    toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.show()
    return toggle_selector


class RectangleSelector(widgets.RectangleSelector):
    def get_lims(self, scale = 1):
        extent = self.extents
        return np.asarray(extent)*scale


def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    print(" The button you used were: %s %s" % (eclick.button, erelease.button))


def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


        
def draw_rectangle_selector(im):
    fig, current_ax = plt.subplots()                 # make a new plotting range
    plt.imshow(im)
    print("\n      click  -->  release")

    # drawtype is 'box' or 'line' or 'none'
    toggle_selector.RS = RectangleSelector(current_ax, line_select_callback,
                                           drawtype='box', useblit=True,
                                           button=[1, 3],  # don't use middle button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.show()
    return toggle_selector





def add_annotation_markers(file):
    '''
    Finds and plots the annotations from dm3/dm4 file metadata as hyperspy markers
    
    file (str): path to dm3/dm4 file
    
    returns: hyperspy.signal with annotations
    '''
    
    
    p3 = hs.load(file)
    
    dm4file = ncempy.io.dm.fileDM(file)

    tag_dict = dm4file.allTags

    keys = list(dm4file.allTags.keys())

    text_ann_keys = [k for k in keys if k.find('AnnotationType') != -1 and np.all(dm4file.allTags[k] == 13)]

    txt = []
    pos = []

    for text_ann_k in text_ann_keys:

        txt.append(tag_dict[text_ann_k.strip('AnnotationType') + 'Text'])

        txt_rect = tag_dict[text_ann_k.strip('AnnotationType') + 'Rectangle']

        pos.append(txt_rect)


    for i, t in enumerate(txt):
        c = pos[i]
        lx, rx = p3.axes_manager[0].scale * c[1], p3.axes_manager[0].scale * c[3]
        ly, ry = p3.axes_manager[1].scale * c[0], p3.axes_manager[1].scale * c[2]
        px= np.mean((lx,rx))
        py = np.mean((ly,ry))

        m = hs.plot.markers.text(px,py, t, color = 'red', fontsize = 20, ha = 'center')
        p3.add_marker(m, permanent=True)


    box_ann_keys = [k for k in keys if k.find('AnnotationType') != -1 and np.all(dm4file.allTags[k] == 5)]

    rec_coords = []

    for box_ann_k in box_ann_keys:

        rec_coords.append(tag_dict[box_ann_k.strip('AnnotationType') + 'Rectangle'])

    for c in rec_coords:
        lx, rx = p3.axes_manager[0].scale * c[1], p3.axes_manager[0].scale * c[3]
        ly, ry = p3.axes_manager[1].scale * c[0], p3.axes_manager[1].scale * c[2]
        m = hs.plot.markers.rectangle(lx,ly,rx,ry, color = 'red')
        p3.add_marker(m, permanent=True)
    
    return p3

def get_img_grid_array(shape, flatten = True):
    ps = shape
    xgrid = np.repeat(np.arange(0, ps[0])[:,None], ps[1], axis = 1)[:,:,None]
    ygrid = np.repeat(np.arange(0, ps[1])[None,:], ps[0], axis = 0)[:,:,None]
    if flatten == False:
        return np.concatenate((xgrid, ygrid), axis = 2)
    else:
        return np.concatenate((xgrid, ygrid), axis = 2).reshape((ps[0]*ps[1],2))

#replaced with untested GPT formula
#def get_minmax_grid_array(info, flatten = True):
#    '''
#    info: ((min_val1, max_val1, npoints1), (min_val2, max_val2, npoints2))
#    '''
#    min_val1, max_val1, npoints1 = info[0]
#    min_val2, max_val2, npoints2 = info[1]
#    xgrid = np.repeat(np.linspace(min_val1, max_val1, npoints1)[:,None], npoints2, axis = 1)[:,:,None]
#    ygrid = np.repeat(np.linspace(min_val2, max_val2, npoints2)[None,:], npoints1, axis = 0)[:,:,None]
#    if flatten == False:
#        return np.concatenate((xgrid, ygrid), axis = 2)
#    else:
#        return np.concatenate((xgrid, ygrid), axis = 2).reshape((npoints1*npoints2,2))


from numba import njit

@njit
def get_minmax_grid_array_flatten(info):
    '''
    info: ((min_val1, max_val1, npoints1), (min_val2, max_val2, npoints2))
    '''
    min_val1, max_val1, npoints1 = info[0]
    min_val2, max_val2, npoints2 = info[1]

    xgrid = np.linspace(min_val1, max_val1, npoints1)
    ygrid = np.linspace(min_val2, max_val2, npoints2)

    res = np.empty((npoints1 * npoints2, 2), dtype=np.float64)
    for i in range(npoints1):
        for j in range(npoints2):
            res[i * npoints2 + j] = xgrid[i], ygrid[j]
    return res

@njit
def get_minmax_grid_array_no_flatten(info):
    '''
    info: ((min_val1, max_val1, npoints1), (min_val2, max_val2, npoints2))
    '''
    min_val1, max_val1, npoints1 = info[0]
    min_val2, max_val2, npoints2 = info[1]

    xgrid = np.linspace(min_val1, max_val1, npoints1)
    ygrid = np.linspace(min_val2, max_val2, npoints2)

    res = np.empty((npoints1, npoints2, 2), dtype=np.float64)
    for i in range(npoints1):
        for j in range(npoints2):
            res[i, j] = xgrid[i], ygrid[j]
    return res

def get_minmax_grid_array(info, flatten=True):
    if flatten:
        return get_minmax_grid_array_flatten(info)
    else:
        return get_minmax_grid_array_no_flatten(info)

