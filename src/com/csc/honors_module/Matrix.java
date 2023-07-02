package com.csc.honors_module;

import java.util.function.BiConsumer;
import java.util.function.Function;

import org.opencv.core.Mat;

public class Matrix extends Mat {
	Mat internal;
	
	public static Matrix create_from(Mat matrix) {
		Matrix instance = new Matrix();
		instance.internal = matrix;
		return instance;
	}
	
	public void for_each(BiConsumer<Integer, Integer> function) {
		for (int x = 0; x < internal.rows(); ++x) {
			for (int y = 0; y < internal.cols(); ++y) {
				function.accept(x, y);
			}
		}
	}
	
	public static void replace_at(Mat matrix, int[] coordinates, Function<Double, Double> replacement) {
		double[] element = matrix.get(coordinates[0], coordinates[1]);
		element[0] = replacement.apply(element[0]);
		matrix.put(coordinates[0], coordinates[1], element);
	}
	
	public static void transform_using(Mat matrix, Function<Double, Double> replacement) {
		Matrix.create_from(matrix).for_each((x, y) -> {
			Matrix.replace_at(matrix, new int[] {x, y}, (element) -> {
				return replacement.apply(element);
			});
		});

	}
}
