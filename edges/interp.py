#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:56:13 2017

@author: liuhantian
"""

import numpy as np

#####
#   author: Xiao Zhang
#
#   Function Input 
#   x     M*N            x coordinates meshgrid
#   y     M*N            y coordinates meshgrid
#   v     M*N            the value lies on grid point which is corresponding to the meshgrid coordinates 
#   xq    M1*N1 or M2    the query points x coordinates
#   yq    M1*N1 or M2    the query points y coordinates
#         
##########
#   Function Output
#   interpv , the interpolated value at querying coordinates xq, yq, it has the same size as xq and yq.
##########
#   For project 1, x and y should be the meshgrid of Mag
#                  v = Mag
#                  xq and yq are the coordinates of the interpolated location, i.e the coordinates computed based on the gradient orientation.

def interp2d(x, y, v, xq, yq):

	x = x.astype(np.int32)
	y = y.astype(np.int32)

	dim_input = 1
	if len(xq.shape) == 2 or len(yq.shape) == 2:
		if xq.shape != yq.shape:
			raise 'Xq Yq should have the same shape'
		dim_input = 2
		q_h = xq.shape[0]
		q_w = xq.shape[1]
		xq = xq.flatten()
		yq = yq.flatten()

	h = x.shape[0]
	w = x.shape[1]
	if x.shape != y.shape:
		raise 'the shape of grid coordinate X and grid coordinate Y doesnt match'
	if y.shape != v.shape:
		raise 'the shape of grid coordinate and value V doesnt match'
	if xq.shape != yq.shape:
		raise 'query coordinates Xq Yq should have same shape'


	x_floor = np.floor(xq).astype(np.int32)
	y_floor = np.floor(yq).astype(np.int32)
	x_ceil = np.ceil(xq).astype(np.int32)
	y_ceil = np.ceil(yq).astype(np.int32)

	x_floor[x_floor<0] = 0
	y_floor[y_floor<0] = 0
	x_ceil[x_ceil<0] = 0
	y_ceil[y_ceil<0] = 0

	x_floor[x_floor>=w-1] = w-1
	y_floor[y_floor>=h-1] = h-1
	x_ceil[x_ceil>=w-1] = w-1
	y_ceil[y_ceil>=h-1] = h-1

	v1 = v[y_floor, x_floor]
	v2 = v[y_floor, x_ceil]
	v3 = v[y_ceil, x_floor]
	v4 = v[y_ceil, x_ceil]

	lh = yq - y_floor
	lw = xq - x_floor
	hh = 1 - lh
	hw = 1 - lw

	w1 = hh * hw
	w2 = hh * lw
	w3 = lh * hw
	w4 = lh * lw

	interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

	if dim_input == 2:
		return interp_val.reshape(q_h,q_w)
	return interp_val


if __name__ == "__main__":
	print('demo of the interp2 function')
	x_mesh, y_mesh = np.meshgrid(np.arange(4),np.arange(4))
	print('x, the x meshgrid:')
	print(x_mesh)
	print('y, the y meshgrid:')
	print(y_mesh)
	v = np.arange(16).reshape(4,4)
	print('v, the value located on the coordinates above')
	print(v)
	xq_mesh, yq_mesh = np.meshgrid(np.arange(0,3.5,0.5),np.arange(0,3.5,0.5))
	print('xq_mesh, the query points x coordinates:')
	print(xq_mesh)
	print('yq_mesh, the query points y coordinates:')
	print(yq_mesh)
	interpv = interp2d(x_mesh,y_mesh,v,xq_mesh,yq_mesh)
	print('output the interpolated point at query points, here we simply upsample the original input twice')
	print(interpv)
