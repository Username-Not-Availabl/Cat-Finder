/**
 * 
 */
package com.csc.honors_module;

import java.io.Serializable;
import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.csc.honors_module.MathUtils.Activation;
import com.csc.honors_module.MathUtils.Random;
import com.csc.honors_module.Model.Initialization;
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
	public static class Convolution2D extends Layer implements CanHaveActivationFunction {
		
//		private int depth; 			// [output.depth]
//		private int input_depth;    // [input.depth]
		
//		private int[] input_shape;  // [input.shape]
//		private int[] output_shape; // [output.shape]

		private int[] kernelShape;
		
//		private Mat[][] kernels;
		private Mat[] kernels;
		private Mat[] biases;
//		private Mat biases;
		
//		private Mat input;  // [tensor]
//		private Mat output; // [tensor]
		
//		TODO: should be private: 
//		is only public for debug purposes
		public static class Self {
			private int depth;
			private int[] dimensions;
			private Mat tensor;
			public int channels;
			public int width;
			public int height;
		}
		
		private Convolution2D.Self input;
		private Convolution2D.Self output;

		private Supplier<Activation> activation_function;

		private int[] stride;

		private int[] padding;

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
		
		//SECOND
//		public Convolution2D(Convolution2D.Self input, Convolution2D.Self output, int[] kernels_shape, Mat[][] kernels, Mat biases, Supplier<Activation> activation_function) {
//			this.input = input;
//			this.output = output;
//			
//			this.kernels = kernels;
//			this.kernels_shape = kernels_shape;
//			
//			this.biases = biases;
//			
//			this.activation_function = activation_function;
//		}


//		public Convolution2D(Self input, Self output, int[] kernelSize, Mat[] kernels, Mat[] biases, int stride, int padding, Supplier<Activation> activation_function) {
//			// TODO Auto-generated constructor stub
//			this.input = input;
//			this.output = output;
//			
//			this.kernelShape = kernelSize;
//			this.kernels = kernels;
//			this.biases = biases;
//			
////			this.stride = stride;
////			this.padding = padding;
//			
//			this.activation_function = activation_function;
//		}

		public Convolution2D(Self input, Self output, int[] kernelSize, Mat[] kernels, Mat[] biases, int[] stride, int[] padding, Supplier<Activation> activation_function) {
			// TODO Auto-generated constructor stub
			this.input = input;
			this.output = output;

			this.kernelShape = kernelSize;
			this.kernels = kernels;
			this.biases = biases;

			this.stride = stride;
			this.padding = padding;

			this.activation_function = activation_function;

		}

		//		@Override
//		public Mat forward_propogate_with(Mat input) {
////			TODO: Probably check if the input tensor fits the saved dimensions
//			input.convertTo(input, CvType.CV_32FC3);
//			this.input.tensor = input;
//			this.output.tensor = this.biases.clone();
//			Debug.print(this.output.tensor.cols(), this.output.tensor.cols());
//			
////			Debug.printSurrounded("input", input, Debug::print3DM);
////			Debug.printSurrounded("output", this.output.tensor, Debug::print3DM);
//			for (int x = 0; x < this.output.depth; ++x) {
//				for (int y = 0; y < this.input.depth; ++y) {
//					Mat reshaped = this.input.tensor.reshape(1, this.input.depth);
//					
//					Mat slice = reshaped.row(x).reshape(1, this.input.dimensions[1]);					
//					Mat convolved = new Mat();
//					Mat kernel = this.kernels[x][y];
//					Imgproc.filter2D(slice, convolved, -1, kernel);
////					Imgproc.matTemplate(slice, kernel, convolved, Imgproc.TM_CCORR)
//					
//					int output_width = this.output.dimensions[2];
//					int output_height = this.output.dimensions[1];
//					Mat subsection = convolved.submat(new Rect(0, x, output_width, output_height));
////					Debug.print(subsection.type(), this.output.tensor.type());
//					
//					Core.add(
//							this.output.tensor.reshape(1, this.output.depth).row(x).reshape(1, this.output.dimensions[1]), 
//							subsection, 
//							this.output.tensor.reshape(1, this.output.depth).row(x).reshape(1, this.output.dimensions[1])
//							);
//					
//				}
//			}
//			return this.output.tensor; // NOTE: maybe return clone instead
//		}
//		@Override
//		public Mat forward_propogate_with(Mat input) {
//			this.input.tensor = input;
//			this.output.tensor = new Mat(this.output.height, this.output.width, CvType.CV_32FC3);
//			List<Mat> biases = Arrays.stream(this.biases).collect(Collectors.toCollection(ArrayList::new));
//			Core.hconcat(biases, this.output.tensor);
//			
//			ArrayList<Mat> output = new ArrayList<Mat>();
//			output.ensureCapacity(kernels.length);
//			for (int i = 0; i < kernels.length; ++i) {
//				output.add(new Mat(this.output.height, this.output.width, CvType.CV_32FC1)); 
////				Imgproc.filter2D(input.reshape(1), input.reshape(1), -1, kernels[i].reshape(1));
//				Imgproc.filter2D(input, output.get(i), -1, kernels[i].reshape(1));
//			}
////			this.output.ten
////			Core.vconcat(output, this.output.tensor);
//			Core.hconcat(output, this.output.tensor);
////			return this.output.tensor;
//			Debug.prints("return_");
////			Debug.print(this.output.height, this.output.width);
////			Debug.print(this.output.tensor.submat(0, 5, 0, 5));
////			this.output.tensor = this.output.tensor.n_reshape(1, int[] {this.output.height, this.output.width});
//			this.output.tensor = this.output.tensor.reshape(this.output.channels, new int[] {this.output.height, this.output.width / this.output.height});
//			return this.output.tensor.clone();
//		}
		@Override
		public Mat forward_propogate_with(Mat input) {
			this.input.tensor = input;
			this.output.tensor = new Mat(this.output.height, this.output.width, CvType.CV_32FC3);

//			Mat output = Imgproc.convertMaps(flattened, flattened, input, flattened, 0);
//			this.output.tensor = new Mat(output.dimensions, input.type());
			this.output.tensor = Mat.zeros(output.dimensions, CvType.CV_32F + (8 * (input.channels() - 1 - 1)) );
//			Debug.printNamed(
//					input.channels() - 1, CvType.CV_32F + (8 * (input.channels() - 2)), 
//					this.output.tensor
//					);
			
//			Debug.printNamed(input.channels(), this.output.tensor.channels());
			
//			for (int i = 0; i < input.channels(); ++i) {
			List<Mat> e = new ArrayList<>();
			for (int i = 0; i < kernels.length; ++i) {
//				Debug.print(output.tensor.submat(i, i + 2, i, i + 2));
//				Mat r = new Mat(2, 2, CvType.CV_32F + (8 * (input.channels() - 2)));
				Mat r = new Mat();
//				Debug.print(r);
//				Imgproc.matchTemplate(input, kernels[i], output.tensor.submat(i, i + 1, i, i + 1), Imgproc.TM_CCORR);
//				Imgproc.matchTemplate(input, kernels[i], output.tensor.row(i), Imgproc.TM_CCORR);
//				Imgproc.matchTemplate(input, kernels[i], output.tensor, Imgproc.TM_CCORR);
//				Imgproc.matchTemplate(input, kernels[i], r, Imgproc.TM_CCORR);
//				Mat submat = input.reshape(1).row(i).reshape(1, new int[]{3, 3});
				Mat submat = input.reshape(1).row(i).reshape(1, new int[]{this.input.height, this.input.width});
//				Debug.print(submat.);
				List<Mat> channels = new ArrayList<>();
				Core.split(kernels[i], channels);
				Mat kernel = channels.get(0);
//				Mat kernel = kernels[i].reshape(1);
//				Debug.print(kernel);
				Imgproc.matchTemplate(submat, kernel, r, Imgproc.TM_CCORR);
//				Imgproc.matchTemplate(input.reshape(1).row(i).reshape(1, new int[]{3, 3}), kernels[i], r, Imgproc.TM_CCORR);
//				Core.add(input.row(i), biases[i], input.row(i));
//				Debug.printNamed(input.row(i));
//				Debug.print(r, kernels[i], input);
				Debug.print(r);
				Core.add(r, biases[i], r);
				e.add(r);
//				Debug.print(output.tensor.submat(i, i + 1, i, i + 1));
			}
			
			
//			Mat s = new Mat();
//			List<Mat> b = Arrays.stream(this.kernels).collect(Collectors.toCollection(ArrayList::new));
//			Core.hconcat(b, s);
////			s = s.reshape(3, kernelShape)
//			Mat r = new Mat();
//			Debug.print(s);
//			Imgproc.matchTemplate(input, s, r, Imgproc.TM_CCORR);
//			Debug.print(r);
//			Debug.print(input.reshape(1).row(0).reshape(1, new int[]{3, 3}));
//			Debug.print();
			
			Core.merge(e, this.output.tensor);
////			Debug.printNamed(output.dimensions);
////			for (int i = 0; i < biases.length; ++i) {
//////				Core.add(output.tensor, input, input);
//////				input.s
////			}
//			Debug.print(output.tensor.channels(), output.tensor);
			this.output.channels = this.output.tensor.channels();
			return this.output.tensor.clone();
		}

		
		@Override
		public Mat backward_propogate_with(Mat outputGradient, double learning_rate) {
			
////			All in singular
////			NOTE: [(delta{E}/delta{K})(Kernel Gradient) = X(input) <crosscorrelatedwith> (delta{E}/delta{Y})(Output Gradient)]
//			Mat kernelGradient = Mat.zeros(kernelShape, kernels[0].type());
//			Mat submat2 = input.tensor.reshape(1).row(0).reshape(1, new int[]{this.input.height, this.input.width});
//			List<Mat> outputGradients2 = new ArrayList<>();
//			Core.split(outputGradient, outputGradients2);
//			Imgproc.matchTemplate(submat2, outputGradients2.get(0), kernelGradient, Imgproc.TM_CCORR);
//			
////			NOTE: [(delta{E}/delta{Y})(Bias Gradient) = (delta{E}/delta{Y})(Output Gradient)]
//			Mat biasGradient2 = outputGradient;
//			
////			NOTE: [(delta{E}/delta{X})(Input Gradient) = (delta{E}/delta{Y})(Output Gradient) <convolvedwith> K(kernel)]
//			Mat inputGradient2 = new Mat();
//			List<Mat> kernels = new ArrayList<Mat>();
//			Core.split(this.kernels[0], kernels);
//			Mat paddedOutputGradientsSlice = new Mat();
//			Core.copyMakeBorder(outputGradients2.get(0), paddedOutputGradientsSlice, 0, 1, 0, 1, Core.BORDER_CONSTANT, Scalar.all(0));
//			Mat paddedKernel = new Mat();
//			Core.copyMakeBorder(kernels.get(0), paddedKernel, 0, 1, 0, 1, Core.BORDER_CONSTANT, Scalar.all(0));
////			Imgproc.filter2D(outputGradients.get(0), inputGradient, CvType.CV_32F, kernels.get(0), new Point(-1, -1), 0, Core.BORDER_CONSTANT);
//			Imgproc.filter2D(paddedOutputGradientsSlice, inputGradient2, CvType.CV_32F, paddedKernel, new Point(-1, -1), 0, Core.BORDER_CONSTANT);
//			
//			Debug.print(kernelGradient);
//			Debug.print(outputGradient);
//			Debug.print(inputGradient2);
			
//			Mat kernelGradient = 
			List<Mat> outputGradientSlices = new ArrayList<>();
			Core.split(outputGradient, outputGradientSlices);
			ArrayList<Mat> kernelGradientSlices = new ArrayList<>();
			for (int i = 0; i < this.kernels.length; ++i) {
				Mat submat = input.tensor.reshape(i).row(0).reshape(1, new int[]{this.input.height, this.input.width});
				Mat kernelGradientSlice = new Mat();
				Imgproc.matchTemplate(submat, outputGradientSlices.get(i), kernelGradientSlice, Imgproc.TM_CCORR);
				kernelGradientSlices.add(kernelGradientSlice);
			}
			
			Mat biasGradient = outputGradient;
			
			ArrayList<Mat> inputGradientSlices = new ArrayList<>();
			for (int i = 0; i < this.output.channels; ++i) {
				List<Mat> kernel = new ArrayList<Mat>();
				Core.split(this.kernels[i], kernel);
				
				Mat convolutionSum = new Mat(Arrays.copyOfRange(this.input.dimensions, 0, 2), CvType.CV_32F);
				for (int j = 0; j < this.input.channels; ++j) {
					Mat inputGradientSlice = new Mat();
					
					Mat paddedOutputGradientSlice = new Mat();
					Debug.print(this.input.channels, outputGradientSlices.size());
//					:Point-of-Error:v
					Core.copyMakeBorder(outputGradientSlices.get(j), paddedOutputGradientSlice, 0, 1, 0, 1, Core.BORDER_CONSTANT, Scalar.all(0));
					
					Mat paddedKernelSlice = new Mat();
					Core.copyMakeBorder(kernel.get(j), paddedKernelSlice, 0, 1, 0, 1, Core.BORDER_CONSTANT, Scalar.all(0));
					
					Imgproc.filter2D(paddedOutputGradientSlice, inputGradientSlice, CvType.CV_32F, paddedKernelSlice, new Point(-1, -1), 0, Core.BORDER_CONSTANT);
					Core.add(inputGradientSlice, convolutionSum, convolutionSum);
				}
				inputGradientSlices.add(convolutionSum);
			}
			
			
			for (int i = 0; i < this.kernels.length; ++i) {
				Core.multiply(kernelGradientSlices.get(i), new Scalar(learning_rate), kernelGradientSlices.get(i));
				ArrayList<Mat> kernel = new ArrayList<>();
				Core.split(this.kernels[i], kernel);
				Core.subtract(kernel.get(0), kernelGradientSlices.get(i), kernel.get(0));
			}
			
			for (int i = 0; i < this.kernels.length; ++i) {
				Core.multiply(outputGradientSlices.get(i), new Scalar(learning_rate), outputGradientSlices.get(i));
				Core.subtract(this.biases[i], outputGradientSlices.get(i), this.biases[i]);
			}
			
			Mat inputGradient = new Mat();
			Core.merge(inputGradientSlices, inputGradient);
			return inputGradient.clone();
		}
		
////////////////////////////
////		https://medium.com/geekculture/building-deep-convolutional-neural-networks-from-scratch-in-java-583a780b56f2
////		https://github.com/eliasyilma/CNN
//		@Override
//		public Mat backward_propogate_with(Mat output_gradient, double learning_rate) {
////			NOTE: Maybe make kernelsGradient also a 2D Mat Array
////			Mat[][] kernelsGradient;
//			Mat kernelsGradient = Mat.zeros(
//					this.kernels_shape[0],
//					this.kernels_shape[1] * this.kernels_shape[2] * this.kernels_shape[3], 
//					CvType.CV_32FC1
//			);
////for (int x = 0; x < kernelsGradient.size(0); ++x) {
////	for (int y = 0; y < kernelsGradient.size(1); ++y) {
////		int[] index = { x, y };
////		kernelsGradient.put(index, x * kernelsGradient.size(1) + y);
////	}
////}
//			
//			Mat inputGradient = Mat.zeros(
//					this.input.dimensions[0],
//					this.input.dimensions[1] * this.input.dimensions[2], 
//					CvType.CV_32FC1
//			);
//			
//			for (int x = 0; x < this.output.depth; ++x) {
//				for (int y = 0; y < this.input.depth; ++y) {
//					Mat kernelGradient = kernelsGradient.row(x).reshape(1, new int[] {this.kernels_shape[1], this.kernels_shape[2], this.kernels_shape[3]});
//					
//					Mat inputSubmat = this.input.tensor
//												.reshape(1, this.input.depth)
//												.row(x)
//												.reshape(1, this.input.dimensions[1]);
//					Mat outputSubmat = this.output.tensor
//												 .reshape(1, this.output.depth)
//												 .row(x)
//												 .reshape(1, this.output.dimensions[1]);
//					Imgproc.filter2D(inputGradient, kernelGradient, -1, outputSubmat, new Point(0, 0), Core.BORDER_DEFAULT);
////					Imgproc.matTemplate(inputSubmat, outputSubmat, kernelGradient, Imgproc.TM_CCORR)
//					
//					Mat rotatedKernel = new Mat();
//					Core.rotate(this.kernels[x][y], rotatedKernel, Core.ROTATE_180);
//					Imgproc.filter2D(outputSubmat, inputSubmat, -1, rotatedKernel);
//				}
//			}
//			
//			Core.multiply(kernelsGradient, new Scalar(learning_rate), kernelsGradient);
//			for (int x = 0; x < this.output.depth; ++x) {
//				for (int y = 0; y < this.input.depth; ++y) {
//					Core.subtract(
//							kernels[x][y], 
//							kernelsGradient.row(x).reshape(1, kernels_shape[1]).row(y).reshape(1, kernels_shape[2]), 
//							kernels[x][y]
//					);
//				}
//			}
//			
//			Core.multiply(output_gradient.reshape(1, 1), new Scalar(learning_rate), output_gradient.reshape(1, 1));
////Debug.printSurrounded("biases", this.biases, Debug::print3DM);			
////Debug.printSurrounded("biases", this.biases.reshape(1, 1), Debug::print3DM);			
////Debug.printSurrounded("output_gradient", output_gradient, Debug::print3DM);			
//			Core.subtract(this.biases.reshape(1, 1), output_gradient.reshape(1, 1), this.biases.reshape(1, 1));
//			return inputGradient.reshape(1, this.input.dimensions);
//		}

//////////////////////////
//		TODO: add to Layer class to be implemented by all layers
//		private static double calculate_output_size() {
//			return 0;
//		}

		@Override
		public Supplier<Activation> activation() {
			return this.activation_function;
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

	
//	NOTE: filters => "Integer, the dimensionality of the output space (i.e. the number)"
	public static Convolution2D Convolution2D(int filters, int[] kernelSize, int[] input_shape, int channels, Supplier<Activation> activation_function) {
		int in_channels = input_shape[0];
		int in_height = input_shape[1];
		int in_width = input_shape[2];
		
		Convolution2D.Self input = new Convolution2D.Self();
		input.channels = in_channels;
		input.height = in_height;
		input.width = in_width;
		input.dimensions = input_shape;
		
		Convolution2D.Self output = new Convolution2D.Self();
//		output.channels = depth;
		
//		output.height = in_height - kernelSize[0] + 1;
//		output.width = in_width - kernelSize[1] + 1;
		
//		https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d
		int[] padding = new int[] {0, 0};
		int[] stride = new int[] {1, 1};
		int[] dilation = new int[] {1, 1};
//		TODO: move to method/function in a useful way
//		output.height = (in_height - kernelSize[0] + (2 * padding)) / stride + 1;
//		output.width = (in_width - kernelSize[1] + (2 * padding)) / stride + 1;
///////////////////
//		output.height = (in_height - kernelSize[0] + (2 * padding[0]) + (kernelSize[0] - 1)) / stride[0] + 1;
		output.height = (in_height + (2 * padding[0]) - (dilation[0] * (kernelSize[0] - 1)) - 1) / stride[0] + 1;
//		int a = (in_height + (2 * padding[0]) + dilation[0] * (kernelSize[0] - 1));
//		Debug.printNamed(in_height, (2 * padding[0]), dilation[0], (kernelSize[0] - 1), a);
//		output.width = (in_width - kernelSize[1] + (2 * padding[1]) + (kernelSize[1] - 1)) / stride[1] + 1;
		output.width = (in_width + (2 * padding[1]) - (dilation[1] * (kernelSize[1] - 1)) - 1) / stride[1] + 1;
//		int[] dilation = new int[] {0, 0};
//		double numerator = (in_height - (2 * padding) - dilation[0] * (kernelSize[0] - 1) - 1);
//		output.height = (int) Math.floor( ( numerator / stride) + 1 );
//		Debug.printNamed(numerator, output.height);
//		numerator = (in_width - (2 * padding) - dilation[1] * (kernelSize[1] - 1) - 1);
//		output.width = (int) Math.floor( ( numerator / stride) + 1 );
//		output.depth = filters; //NOTE: usefulness is questionable
//		Debug.printNamed(numerator, output.height);
//////////////////////////////////
//		Debug.printNamed(in_height, in_width, in_channels);
//		Debug.printNamed(input.height, input.width, in_channels);
//		Debug.printNamed(output.height, output.width);
		
//		1 = Math.floor(in_height + (2 * padding) - dilation[0])
		output.dimensions = new int[] {output.height, output.width};
		
		
//		output.dimensions = new int[] {depth , in_height - kernelSize[0] + 1, in_width - kernelSize[1] + 1 };

//		int[] kernels_shape = new int[] { depth, in_channels, kernel_size[0], kernel_size[1] };
//		Mat[][] kernels = new Mat[depth][in_channels];
//		for (int x = 0; x < depth; ++x) {
//			for (int y = 0; y < in_channels; ++y) {
//				kernels[x][y] = new Mat(kernel_size[0], kernel_size[1], CvType.CV_32FC1);
//				Core.randn(kernels[x][y], 0, 1);
//			}
//		}
		Mat[] kernels = new Mat[filters];
		Mat biases[] = new Mat[filters];
		for (int i = 0; i < filters; ++i) {
//			 TODO: allow user to instantiate with any number of channels
//					CvType.CV_32F + (8 * in_channels)
//					https://gist.github.com/yangcha/38f2fa630e223a8546f9b48ebbb3e61a
			kernels[i] = new Mat(kernelSize[0], kernelSize[1], CvType.CV_32F + (8 * (in_channels - 1)) );
			Model.initialize_with(kernels[i], kernelSize, new Initialization.Xavier());

			biases[i] = new Mat(output.height, output.width, CvType.CV_32F + 0 );
			Model.initialize_with(biases[i], new int[] {input.height, input.width}, new Initialization.GlorotUniform());

//		::TEST
			
			Model.initialize_with(kernels[i], kernelSize, new Initialization() {

				@Override
				protected void implementation(Mat kernel, int[] kernelSize) {
					// TODO Auto-generated method stub
//					kernel.setTo(new Scalar(1));
					Matrix.create_from(kernel)
					  	.for_each((x, y) -> {
						  kernel.put(new int[] {x, y}, 1, 1, 1);
					  });
				}
				
			});
			
			Model.initialize_with(biases[i], new int[] {input.height, input.width}, new Initialization() {

				@Override
				protected void implementation(Mat kernel, int[] kernelSize) {
					// TODO Auto-generated method stub
					kernel.setTo(new Scalar(1));
				}
				
			});
//			int index = 0;
//			for (int a = 0; a < kernels[i].size(0); ++a) {
//				for (int b = 0; b < kernels[i].size(1); ++b) {
////					for (int c = 0; c < kernels[i].channels(); ++c) {
////						coordinates[0] + (width * (coordinates[1] + (height * coordinates[2])))
////						int index = a + (kernelSize[1] * (b + 0));
////						Debug.print(index, new int[] {a, b});
//						
//						kernels[i].put(new int[] {a, b}, index++, index++, index++);
////						Debug.print(a, b);
////					}
//				}
//			}
//    ::TEST::END
		}
//		::TEST
//		Debug.print(kernels); 
//		::TEST::END
		
//		output.depth = kernels.length;
		
//		Debug.print(filters);
		
//		NOTE: Could possibly be changed to be of type Mat[]
//		Mat biases = new Mat(output.dimensions, CvType.CV_32FC1);	
//		Mat biases = new Mat(output.dimensions, CvType.CV_32FC3);

//		{NOTE:} Rewrite from here
//		depth = in_channels
//		https://gist.github.com/yangcha/38f2fa630e223a8546f9b48ebbb3e61a
//		int channels = output.depth;
//		Mat biases = new Mat(new int[]{output.dimensions[1], output.dimensions[2]}, CvType.CV_32F + (8 * channels));	
//		Core.randn(biases, 0, 1);
		
		
		return new Convolution2D(input, output, kernelSize, kernels, biases, stride, padding, activation_function);
	}
	
//	public static Convolution2D Convolution2D(int[] input_shape, int kernel_size, int depth, Supplier<Activation> activation_function) {
//		int input_depth = input_shape[0];
//		int input_height = input_shape[1];
//		int input_width = input_shape[2];
//		
//		Convolution2D.Self input = new Convolution2D.Self();
//		input.depth = input_depth;
//		input.dimensions = input_shape;
//		
//		Convolution2D.Self output = new Convolution2D.Self();
//		output.depth = depth;
//		output.dimensions = new int[] { depth, input_height - kernel_size + 1, input_width - kernel_size + 1 };
//
//		int[] kernels_shape = new int[] { depth, input_depth, kernel_size, kernel_size };
//		Mat[][] kernels = new Mat[depth][input_depth];
//		for (int x = 0; x < depth; ++x) {
//			for (int y = 0; y < input_depth; ++y) {
//				kernels[x][y] = new Mat(kernel_size, kernel_size, CvType.CV_32FC1);
//				Core.randn(kernels[x][y], 0, 1);
//			}
//		}
//		
////		NOTE: Could possibly be changed to be of type Mat[]
////		Mat biases = new Mat(output.dimensions, CvType.CV_32FC1);	
////		Mat biases = new Mat(output.dimensions, CvType.CV_32FC3);
//
////		{NOTE:} Rewrite from here
////		depth = in_channels
////		https://gist.github.com/yangcha/38f2fa630e223a8546f9b48ebbb3e61a
//		int channels = output.depth;
//		Mat biases = new Mat(new int[]{output.dimensions[1], output.dimensions[2]}, CvType.CV_32F + (8 * channels));	
//		Debug.print("biases", biases.cols(), biases.rows());
//		Debug.print("output_dimensions", output.dimensions);
//		Core.randn(biases, 0, 1);
//		
//		return new Convolution2D(input, output, kernels_shape, kernels, biases, activation_function);
//	}

	
//	TODO: Maybe create a class
//	https://pytorch.org/docs/stable/_modules/torch/nn/modules/dropout.html#Dropout
//	https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#dropout
	public static class Dropout extends Layer implements CanHaveActivationFunction {
		private static final long serialVersionUID = -6794876915780198221L;

		private double dropout_ratio;
		private Mat mask;
		private Supplier<Activation> activation_function;
	
		public Dropout(double chance_of_being_zeroed, Supplier<Activation> activation_function) {
			this.dropout_ratio = chance_of_being_zeroed;
			this.activation_function = activation_function;
		}
	
		@Override
		public Mat forward_propogate_with(Mat input) {
			mask = Mat.zeros(input.size(), 0);
			
			Mat result = input.clone();
			Matrix.create_from(result).for_each((x, y) -> {
				Matrix.replace_at(result, new int[] { x, y }, (element) -> {
					if (Random.between(Range.of(infinitum.FALSE, 0.0, 1.0)) >= dropout_ratio) {
						mask.put(x, y, 1);
						return element / (1 - dropout_ratio);
					}
					return 0.0;
				});
			});
			return result;
		}
	
		@Override
		public Mat backward_propogate_with(Mat output_gradient, double learning_rate) {
			Mat scaledGradients = new Mat();
			Core.multiply(output_gradient, new Scalar(1.0 / (1 - dropout_ratio)), scaledGradients);
			
			Mat dropoutGradients = new Mat();
			Core.multiply(scaledGradients, this.mask, dropoutGradients);
			return dropoutGradients;
		}
	
		@Override
		public Supplier<Activation> activation() {
			return this.activation_function;
		}
		
	}
	public static Dropout dropout(double chance_of_being_zeroed, Supplier<Activation> activation_function) {
		if (!Range.of(infinitum.FALSE, 0.0f, 1.0f).contains(chance_of_being_zeroed, true)) {
			throw new InvalidParameterException("Dropout probability umst be contained between Range[0, 1]");
		}
		return new Dropout(chance_of_being_zeroed, activation_function);
	}
	
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
	
	public static class MaxPooling2D extends Layer implements CanHaveActivationFunction {
		private int[] kernel_shape;
		private Mat[][] kernels;
		private int stride;
		
		private static class Self {
			private int depth;
			private int[] dimensions;
			private Mat tensor;
		}
		private MaxPooling2D.Self input;
		private MaxPooling2D.Self output;
		
		private Supplier<Activation> activation_function;

		public MaxPooling2D(MaxPooling2D.Self input, MaxPooling2D.Self output, int[] kernel_shape, Mat[][] kernels, int stride, Supplier<Activation> activation_function) {
			this.input = input;
			this.output = output;
			
			this.kernel_shape = kernel_shape;
			this.kernels = kernels;
			this.stride = stride;
			
			this.activation_function = activation_function;
		}

		@Override
		public Mat forward_propogate_with(Mat input) {
			this.input.tensor = input;
			
			this.output.tensor = new Mat(this.output.dimensions, CvType.CV_32FC1);
			for (int i = 0; i < this.input.depth; ++i) {	
				Mat reshaped = this.input.tensor.reshape(1, this.input.depth);
				Mat slice = reshaped.row(i).reshape(1, this.input.dimensions[1]);
//				
				reshaped = this.output.tensor.reshape(1, this.output.depth);
				Mat output_slice = reshaped.row(i).reshape(1, this.input.dimensions[1]);

//				Imgproc.resize(slice, output_slice, output_slice.size(), 0, 0, Imgproc.INTER_NEAREST);
				Imgproc.resize(slice, output_slice, new Size(), 0, 0, Imgproc.INTER_MAX);
		        Imgproc.resize(output_slice, output_slice, new Size(output.dimensions[2], output.dimensions[1]), 0, 0, Imgproc.INTER_NEAREST);
			}
			return this.output.tensor; // NOTE: maybe return clone instead
		}

		@Override
		public Mat backward_propogate_with(Mat output_gradient, double learning_rate) {
			Mat inputGradient = Mat.zeros(
					this.input.dimensions[0],
					this.input.dimensions[1] * this.input.dimensions[2], 
					CvType.CV_32FC1
			);

			for (int i = 0; i < this.output.depth; ++i) {
				Mat inputSubmat = this.input.tensor
						.reshape(1, this.input.depth)
						.row(i)
						.reshape(1, this.input.dimensions[1]);
				Mat outputSubmat = this.output.tensor
						 .reshape(1, this.output.depth)
						 .row(1)
						 .reshape(1, this.output.dimensions[1]);
		        Imgproc.resize(outputSubmat, inputSubmat, inputSubmat.size(), 0, 0, Imgproc.INTER_NEAREST);
		        Imgproc.resize(inputSubmat, inputSubmat, new Size(inputSubmat.cols(), inputSubmat.rows()), 0, 0, Imgproc.INTER_NEAREST);
			}
			return inputGradient.reshape(1, this.input.dimensions);
		}

		@Override
		public Supplier<Activation> activation() {
			return this.activation_function;
		}
	}

//	https://www.bing.com/search?pglt=41&q=maxpool2d&cvid=a177ebf08c2e43208476c7ccb2cafe8c&aqs=edge.0.0j69i57j0l7.5314j0j1&FORM=ANNTA1&PC=ASTS
	public static MaxPooling2D MaxPooling2D(int[] input_shape, int depth, int kernel_size, int stride, Supplier<Activation> activation_function) {
		int input_depth = input_shape[0];
		int input_height = input_shape[1];
		int input_width = input_shape[2];
		
		MaxPooling2D.Self input = new MaxPooling2D.Self();
		input.depth = input_depth;
		input.dimensions = input_shape;
		
		MaxPooling2D.Self output = new MaxPooling2D.Self();
		output.depth = depth;
		output.dimensions = new int[] { depth, (input_height - kernel_size) / stride + 1, (input_width - kernel_size) / stride + 1 };
		
		int[] kernel_shape = new int[] { depth, input_depth, kernel_size, kernel_size };
		Mat[][] kernels = new Mat[depth][input_depth];
		for (int x = 0; x < depth; ++x) {
			for (int y = 0; y < input_depth; ++y) {
				kernels[x][y] = new Mat(kernel_size, kernel_size, CvType.CV_32FC1);
				Core.randn(kernels[x][y], 0, 1);
			}
		}
		return new MaxPooling2D(input, output, kernel_shape, kernels, stride, activation_function);
	}
	
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
	public static Flatten Flatten() { return null; }
	
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