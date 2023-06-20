package com.csc.honors_module;

import org.opencv.core.Mat;

public class Matrix extends Mat {
	Mat internal;
	
	public static Matrix create(Mat matrix) {
		Matrix instance = new Matrix();
		instance.internal = matrix;
		return instance;
	}
	
	public Matrix() {
		super();
	}
}
