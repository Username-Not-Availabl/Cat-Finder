package com.csc.honors_module;

import java.util.ArrayList;

import com.csc.honors_module.ModelUtils.Either;

public class Neuron {
	public static double minimum_possible_weight;
	public static double maximum_possible_weight;
	
	public double[] weights;
	public double[] cached_weights; // for backward propogation
	
	float gradient;
	float bias;
	float value = 0;
	
	public Neuron(double[] weights, float bias) {
		this.weights = weights;
		this.cached_weights = this.weights;
		
		this.bias = bias;
		this.gradient = 0;
	}
	
	public Neuron(float initial_value) {
		this.weights = null;
		this.cached_weights = this.weights;
		
		this.bias = -1;
		this.gradient = -1;
		this.value = initial_value;
	}
	
	@SuppressWarnings("unchecked")
	public static void setWeightRange(Either<Range<Double>, ArrayList<Double>> range) {
		if (range.get() instanceof Range) {
			Range<Double> _range = ((Range<Double>)range.get());
//			TODO: Maybe add exception if .from or .to are instances of infinitum
			minimum_possible_weight = _range.from.get();
			maximum_possible_weight = _range.to.get();
		}
		
		if (range.get() instanceof ArrayList) {
			ArrayList<Double> _range = ((ArrayList<Double>)range.get());
			minimum_possible_weight = _range.get(0);
			maximum_possible_weight = _range.get(1);
		}
	}
	
//	NOTE: used at the end of the back propogation to switch 
//		  the calculated value ...
	public void regress_to_cached() {this.weights = this.cached_weights;}
}
