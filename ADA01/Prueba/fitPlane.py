#! /usr/bin/env python3
#encoding: utf-8

import numpy as np

from sklearn import linear_model

def fitPlane(x, y, z):
	n=len(x)
	if n != len(y) or n != len(z):
		print('ERROR')
		return
	
	X_data =np.hstack([x.reshape((-1,1)),y.reshape((-1,1))])

	Y_data = z
	
	reg = linear_model.LinearRegression().fit(X_data, Y_data)

	Sol = [reg.coef_[0],reg.coef_[1], 1, reg.intercept_]
	return Sol
