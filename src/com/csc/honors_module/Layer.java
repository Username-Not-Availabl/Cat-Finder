/**
 * 
 */
package com.csc.honors_module;

import java.io.Serializable;
import java.security.InvalidParameterException;
import java.util.function.Supplier;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import com.csc.honors_module.MathUtils.Activation;
import com.csc.honors_module.MathUtils.Random;
import com.csc.honors_module.Range.infinitum;

/**
 * @author mysti
 *
 */
//https://www.geeksforgeeks.org/vector-vs-arraylist-java/
//https://github.com/TheIndependentCode/Neural-Network/blob/master/convolutional.py
public abstract class Layer implements Serializable {
//	public Supplier<Activation> activation_function = Debug.NULL();
	public Neuron[] neurons;
	
	public Layer(int incoming, int contained_neurons) {
		this.neurons = new Neuron[contained_neurons];
		for (int i = 0; i < contained_neurons; ++i) {
			double[] weights = new double[incoming];
			Range<Double> range;
			for (int j = 0; j < incoming; ++j) {
				range = Range.of(infinitum.FALSE, Neuron.minimum_possible_weight, Neuron.maximum_possible_weight);
				weights[j] = MathUtils.extend_by_predicate((Math.random() >= 0.5), -1, Random.between(range));
			}
			range = Range.of(infinitum.FALSE, 0.0, 1.0);
			double random = Random.between(range);
			neurons[i] = new Neuron(weights, (float) MathUtils.extend_by_predicate((Math.random() >= 0.5), -1, random));
		}
	}
	
	abstract public Mat forward_propogate_with(Mat input);
	abstract public Mat backward_propogate_with(Mat output_gradient, double learning_rate);
//	protected abstract Activation activation();
//	private static class Activation extends Layer {
//		
//	}

	public static class Input extends Layer {
		public Input(float[] input) {
//			this.neurons = new Neuron[input.length];
//			for (int i = 0; i < input.length; ++i) {
//				this.neurons[i] = new Neuron(input[i]);
//			}
		}

		@Override
		public Mat forward_propogate_with(Mat input) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public Mat backward_propogate_with(Mat output_gradient, double learning_rate) {
			// TODO Auto-generated method stub
			return null;
		}
	}
//	TODO: take in some non float[] input and 
//	convert it to the float[] required for contructor
	public static Input Input() { return null; }
	
//	https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d
//	https://pytorch.org/docs/stable/generated/torch.ao.nn.quantized.functional.conv2d.html?highlight=conv2d#torch.ao.nn.quantized.functional.conv2d
//	public static class Convolution2D extends Layer {
////		https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv2d
//		private int in_channels;
//		private ArrayList<Integer> _reversed_padding_repeated_twice; 
//		private int out_channels;
//		
//		private ArrayList<Integer> kernel_size;
//		private ArrayList<Integer> stride;
//		private Either<String, ArrayList<Integer>> padding;
//		private ArrayList<Integer> dilation;
//		
//		private Boolean transposed;
//		
//		private ArrayList<Integer> output_padding;
//		private int groups;
//		private String padding_mode;
//		
////		https://pytorch.org/docs/stable/tensors.html#torch.Tensor
//		private Mat weight;
//		private Optional<Mat> bias;
//		
//		private void complete() {
//			this.stride = new ArrayList<Integer>();
//			this.stride.ensureCapacity(1);
//			
//			
//		}
//		
//		public Convolution2D() {}
//		
//		public Object forward_propogate_with(Object input, Mat weight, Optional<Object> bias) {
//			Either<String, ArrayList<Integer>> padding = this.padding;
//			if (this.padding_mode != "zeroes") {
//				input = padded(input, this._reversed_padding_repeated_twice, padding_mode, 0);
//				padding = Either.create_with(List.of(0, 0));
//			}
//			return convolution(input, weight, bias, this.stride, padding, this.dilation, this.groups);
//		}
//
//		private Object padded(Object input, ArrayList<Integer> _reversed_padding_repeated_twice,
//				String padding_mode, float fill_value) {
//			// TODO Auto-generated method stub
//			return null;
//		}
//
////		https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
//		private Object convolution(Object input, Mat weight, Optional<Object> bias, ArrayList<Integer> stride,
//				Either<String, ArrayList<Integer>> padding, ArrayList<Integer> dilation, int groups) {
//			// TODO Auto-generated method stub
////			https://github.com/pytorch/pytorch/blob/v1.13.1/aten/src/ATen/native/Convolution.cpp#L804
////			batchify(input, /*num_spatial_dims*/2, /*func_name*/"conv2d");
//			
//			return null;
//		}
//
//		@Override
//		public Mat forward_propogate_with(Mat input) {
//			// TODO Auto-generated method stub
//			return null;
//		}
//
//		@Override
//		public Mat backward_propogate_with(Mat output_gradient, double learning_rate) {
//			// TODO Auto-generated method stub
//			return null;
//		}
//	}
	public static class Convolution2D extends Layer {
		
//		private int depth; 			// [output.depth]
//		private int input_depth;    // [input.depth]
		
//		private int[] input_shape;  // [input.shape]
//		private int[] output_shape; // [output.shape]

		private int[] kernels_shape;
		
		private Mat[][] kernels;
		private Mat biases;
		
//		private Mat input;  // [tensor]
//		private Mat output; // [tensor]
		
//		TODO: should be private: 
//		is only public for debug purposes
		public static class Self {
			private int depth;
			private int[] dimensions;
			private Mat tensor;
		}
		
		private Convolution2D.Self input;
		private Convolution2D.Self output;

//		public Convolution2D(int depth, int[] input_shape, int input_depth, int[] output_shape, int[] kernels_shape, Mat kernels, Mat biases) {
//			this.depth = depth;
//			
//			this.input_shape = input_shape;
//			this.input_depth = input_depth;
//			
//			this.output_shape = output_shape;
//			this.kernels_shape = kernels_shape;
//			
//			this.kernels = kernels;
//			this.biases = biases;
//		}
		
		public Convolution2D(Convolution2D.Self input, Convolution2D.Self output, int[] kernels_shape, Mat[][] kernels, Mat biases) {
			this.input = input;
			this.output = output;
			
			this.kernels = kernels;
			this.kernels_shape = kernels_shape;
			
			this.biases = biases;
		}

		@Override
		public Mat forward_propogate_with(Mat input) {
//			TODO: Probably check if the input tensor fits the saved dimensions
			this.input.tensor = input;
			this.output.tensor = this.biases.clone();
			
//			Debug.printSurrounded("input", input, Debug::print3DM);
//			Debug.printSurrounded("output", this.output.tensor, Debug::print3DM);
			for (int x = 0; x < this.output.depth; ++x) {
				for (int y = 0; y < this.input.depth; ++y) {
					Mat reshaped = this.input.tensor.reshape(1, this.input.depth);
					
					Mat slice = reshaped.row(x).reshape(1, this.input.dimensions[1]);					
					Mat convolved = new Mat();
					Mat kernel = this.kernels[x][y];
					Imgproc.filter2D(slice, convolved, -1, kernel);
//					Imgproc.matTemplate(slice, kernel, convolved, Imgproc.TM_CCORR)
					
					
					int output_width = this.output.dimensions[2];
					int output_height = this.output.dimensions[1];
					Mat subsection = convolved.submat(new Rect(0, x, output_width, output_height));
										
					Core.add(
							this.output.tensor.reshape(1, this.output.depth).row(x).reshape(1, this.output.dimensions[1]), 
							subsection, 
							this.output.tensor.reshape(1, this.output.depth).row(x).reshape(1, this.output.dimensions[1])
					);
					
				}
			}
			return this.output.tensor;
		}

//		https://medium.com/geekculture/building-deep-convolutional-neural-networks-from-scratch-in-java-583a780b56f2
//		https://github.com/eliasyilma/CNN
		@Override
		public Mat backward_propogate_with(Mat output_gradient, double learning_rate) {
//			NOTE: Maybe make kernelsGradient also a 2D Mat Array
//			Mat[][] kernelsGradient;
			Mat kernelsGradient = Mat.zeros(
					this.kernels_shape[0],
					this.kernels_shape[1] * this.kernels_shape[2] * this.kernels_shape[3], 
					CvType.CV_32FC1
			);
//for (int x = 0; x < kernelsGradient.size(0); ++x) {
//	for (int y = 0; y < kernelsGradient.size(1); ++y) {
//		int[] index = { x, y };
//		kernelsGradient.put(index, x * kernelsGradient.size(1) + y);
//	}
//}
			
			Mat inputGradient = Mat.zeros(
					this.input.dimensions[0],
					this.input.dimensions[1] * this.input.dimensions[2], 
					CvType.CV_32FC1
			);
			
			for (int x = 0; x < this.output.depth; ++x) {
				for (int y = 0; y < this.input.depth; ++y) {
					Mat kernelGradient = kernelsGradient.row(x).reshape(1, new int[] {this.kernels_shape[1], this.kernels_shape[2], this.kernels_shape[3]});
					
					Mat inputSubmat = this.input.tensor
												.reshape(1, this.input.depth)
												.row(x)
												.reshape(1, this.input.dimensions[1]);
					Mat outputSubmat = this.output.tensor
												 .reshape(1, this.output.depth)
												 .row(x)
												 .reshape(1, this.output.dimensions[1]);
					Imgproc.filter2D(inputGradient, kernelGradient, -1, outputSubmat, new Point(0, 0), Core.BORDER_DEFAULT);
//					Imgproc.matTemplate(inputSubmat, outputSubmat, kernelGradient, Imgproc.TM_CCORR)
					
					Mat rotatedKernel = new Mat();
					Core.rotate(this.kernels[x][y], rotatedKernel, Core.ROTATE_180);
					Imgproc.filter2D(outputSubmat, inputSubmat, -1, rotatedKernel);
				}
			}
			
			Core.multiply(kernelsGradient, new Scalar(learning_rate), kernelsGradient);
			for (int x = 0; x < this.output.depth; ++x) {
				for (int y = 0; y < this.input.depth; ++y) {
					Core.subtract(
							kernels[x][y], 
							kernelsGradient.row(x).reshape(1, kernels_shape[1]).row(y).reshape(1, kernels_shape[2]), 
							kernels[x][y]
					);
				}
			}
			
			Core.multiply(output_gradient.reshape(1, 1), new Scalar(learning_rate), output_gradient.reshape(1, 1));
//Debug.printSurrounded("biases", this.biases, Debug::print3DM);			
//Debug.printSurrounded("biases", this.biases.reshape(1, 1), Debug::print3DM);			
//Debug.printSurrounded("output_gradient", output_gradient, Debug::print3DM);			
			Core.subtract(this.biases.reshape(1, 1), output_gradient.reshape(1, 1), this.biases.reshape(1, 1));
			return inputGradient;
		}
	}

//	public static Convolution2D Convolution2D(int[] input_shape, int kernel_size, int depth) {
//		int input_depth = input_shape[0];
//		int input_height = input_shape[1];
//		int input_width = input_shape[2];
//				
//		int[] output_shape = new int[] { depth, input_height - kernel_size + 1, input_width - kernel_size + 1 };
//		int[] kernels_shape = new int[] { depth, input_depth, kernel_size, kernel_size };
//		
//		Mat kernels = new Mat(kernels_shape, CvType.CV_32FC1, new Scalar(0));
//		Core.randn(kernels, 0, 1);
//		
//		Mat biases = new Mat(output_shape, CvType.CV_32FC1, new Scalar(0));
//		Core.randn(biases, 0, 1);
//		
//		return new Convolution2D(depth, input_shape, input_depth, output_shape, kernels_shape, kernels, biases); 
//	}

//	NOTE: presumes intended kernel is of symetrical size
	public static Convolution2D Convolution2D(int[] input_shape, int kernel_size, int depth) {
		int input_depth = input_shape[0];
		int input_height = input_shape[1];
		int input_width = input_shape[2];
		
		Convolution2D.Self input = new Convolution2D.Self();
		input.depth = input_depth;
		input.dimensions = input_shape;
		
		Convolution2D.Self output = new Convolution2D.Self();
		output.depth = depth;
		output.dimensions = new int[] { depth, input_height - kernel_size + 1, input_width - kernel_size + 1 };

		int[] kernels_shape = new int[] { depth, input_depth, kernel_size, kernel_size };
		Mat[][] kernels = new Mat[depth][input_depth];
		for (int x = 0; x < depth; ++x) {
			for (int y = 0; y < input_depth; ++y) {
				kernels[x][y] = new Mat(kernel_size, kernel_size, CvType.CV_32FC1);
				Core.randn(kernels[x][y], 0, 1);
			}
		}
		
//		NOTE: Could possibly be changed to be of type Mat[]
		Mat biases = new Mat(output.dimensions, CvType.CV_32FC1);		
		Core.randn(biases, 0, 1);
		
		return new Convolution2D(input, output, kernels_shape, kernels, biases);
	}
	
//	TODO: Maybe create a class
//	https://pytorch.org/docs/stable/_modules/torch/nn/modules/dropout.html#Dropout
//	https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#dropout
	public static Layer dropout(double chance_of_being_zeroed) {
		if (!Range.of(infinitum.FALSE, 0.0f, 1.0f).contains(chance_of_being_zeroed, true)) {
			throw new InvalidParameterException("Dropout probability umst be contained between Range[0, 1]");
		}
//		TODO[forward?]
		return null;
	}
//	public static Layer dropout() { return null; }
	
	public static class Reshape extends Layer {

		private static class Dimensions {
			private int[] of_input;
			private int[] of_output;
		}
		private Dimensions dimensions;
		
		private static int PLACEHOLDER_CN = 1;
		public Reshape(Reshape.Dimensions dimensions) {
			this.dimensions = dimensions;
		}

		@Override
		public Mat forward_propogate_with(Mat input) {
			return input.reshape(PLACEHOLDER_CN, this.dimensions.of_output);
		}

		@Override
		public Mat backward_propogate_with(Mat output_gradient, double learning_rate) {
			return output_gradient.reshape(PLACEHOLDER_CN, this.dimensions.of_input);
		}

	}

	public static Layer Reshape(int[] input_shape, int[] output_shape) {
		Reshape.Dimensions dimensions = new Reshape.Dimensions();
		dimensions.of_input = input_shape;
		dimensions.of_output = output_shape;
		return new Reshape(dimensions);
	}
	
	public static class MaxPooling2D extends Layer {
		public MaxPooling2D() {}

		@Override
		public Mat forward_propogate_with(Mat input) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public Mat backward_propogate_with(Mat output_gradient, double learning_rate) {
			// TODO Auto-generated method stub
			return null;
		}
	}
	public static Layer MaxPooling2D() { return null; }
	
	public static class Flatten extends Layer {

		@Override
		public Mat forward_propogate_with(Mat input) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public Mat backward_propogate_with(Mat output_gradient, double learning_rate) {
			// TODO Auto-generated method stub
			return null;
		}
	}
	public static Layer Flatten() { return null; }
	
//	https://stackoverflow.com/questions/28901366/opencv-how-do-i-multiply-every-value-of-a-mat-by-a-specified-constant#:~:text=You%20can%20use%20Java%27s%20OpenCv%20Core%20library%3A%20public,mat%29%3B%20%2F%2F%20mat%20values%20are%20all%20255%20now
	public static class Dense extends Layer implements CanHaveActivationFunction {
		public Mat weights; 
		public Mat bias;
		
		private Mat input;
		public Supplier<Activation> activation_function;
		
		public Dense(Mat weights, Mat bias, Supplier<Activation> activation_function) {
			this.weights = weights;
			this.bias = bias;
			this.activation_function = activation_function;
		}
		
		public Mat forward_propogate_with(Mat input) {
			this.input = input;
			Mat result = new Mat(this.bias.rows(), 1, CvType.CV_32FC1);
			Core.gemm(this.weights, input, 1.0, bias, 1.0, result);
			return result;
		}
		
		public Mat backward_propogate_with(Mat output_gradient, double learning_rate) {
			Mat weightsGradient = new Mat();
			Core.gemm(output_gradient, this.input.t(), 1.0, new Mat(), 1.0, weightsGradient);
			
//			NOTE: [output_gradient].rows() should be 1
			Mat inputGradient = new Mat();
			Core.gemm(this.weights.t(), output_gradient, 1.0, new Mat(), 1.0, inputGradient);
			
//			NOTE: Gradient Descent
			Mat destination = new Mat();
			Core.multiply(weightsGradient, new Scalar(learning_rate), destination);
			Core.subtract(this.weights, destination, this.weights);
			
			Core.multiply(output_gradient, new Scalar(learning_rate), destination);
			Core.subtract(this.bias, destination, this.bias);
			return inputGradient;
		}

		public Supplier<Activation> activation() {
			return this.activation_function;
		}
	}
	public static Dense Dense(int[] inout, Supplier<Activation> activation_function) {
		int input_size = inout[0];
		int output_size = inout[1];
		Mat weights = new Mat(output_size, input_size, CvType.CV_32FC1);
		Core.randn(weights, 0, 1);
		
		Mat bias = new Mat(output_size, 1, CvType.CV_32FC1);
		Core.randn(bias, 0, 1);
		
		return new Dense(weights, bias, activation_function); 
	}
	
	protected Layer() {}

	
//	public static void main(String ...args) {
//		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//
////		Layer.Dense layer = Dense(3, 4);
////
////		Mat input = new Mat(3, 1, CvType.CV_32FC1);
////		Core.randn(input, 0, 1);
////
////		Mat result = layer.forward_propogate_with(input);
////
////		input = new Mat(4, 1, CvType.CV_32FC1);
////		Core.randn(input, 0, 1);
////		Mat another = layer.backward_propogate_with(input, 12);
////		Debug.prints(another.dump());
//	}
}

interface CanHaveActivationFunction {
	public Supplier<Activation> activation();
}