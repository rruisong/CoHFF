import numpy as np
from shapely.geometry import Polygon

def index_to_coordinate_3d(i, j, k, x_min, x_max, num_x, y_min, y_max, num_y, z_min, z_max, num_z):
    x_coord = x_min + (i / (num_x - 1)) * (x_max - x_min)
    y_coord = y_min + (j / (num_y - 1)) * (y_max - y_min)
    z_coord = z_min + (k / (num_z - 1)) * (z_max - z_min)
    return x_coord, y_coord, z_coord

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (kent): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,1]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,1]) for i in range(3,-1,-1)] 
    
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])

    voxel_polygon= Polygon(rect1)
    object_polygon = Polygon(rect2)
    inter_area = voxel_polygon.intersection(object_polygon).area
   
    zmin = max(corners1[0,2], corners2[0,2])
    zmax = min(corners1[4,2], corners2[4,2])

    inter_vol = inter_area * max(0.0, zmax-zmin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return vol1, vol2, inter_vol
