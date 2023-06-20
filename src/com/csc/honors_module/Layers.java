/**
 * 
 */
package com.csc.honors_module;

import java.util.ArrayList;
import java.util.Optional;

import org.opencv.core.Mat;

import com.csc.honors_module.ModelUtils.Either;

/**
 * @author mysti
 *
 */
public class Layers {
//	TODO: create new Layer class for this
//	public Neuron[] neurons;
	
	public static class Convolution2D {
//		https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
		private int in_channels;
		private ArrayList<Integer> _reversed_padding_repeated_twice; 
		private int out_channels;
		
		private ArrayList<Integer> kernel_size;
		private ArrayList<Integer> stride;
		private Either<String, ArrayList<Integer>> padding;
		private ArrayList<Integer> dilation;
		private Boolean transposed;
		
		private ArrayList<Integer> output_padding;
		private int groups;
		private String padding_mode;
		
//		https://pytorch.org/docs/stable/tensors.html#torch.Tensor
		private Mat weight;
		private Optional<Mat> bias;
		
		private void complete() {
			this.stride = new ArrayList<Integer>();
			this.stride.ensureCapacity(1);
			
			
		}
		
//		public _conv_forward()
	}
	
	public static Layers dropout(double chance_of_being_zeroed) {
		return null;
	}
	
	public static class MaxPooling2D {
		
	}
	
	public static class Flatten {
		
	}
	
	public static class Dense {
		
	}
}
