package com.csc.honors_module;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import com.csc.honors_module.MathUtils.Activation;
import com.csc.honors_module.MathUtils.Loss;
import com.csc.honors_module.Model.Optimizer;
import com.csc.honors_module.ModelUtils.Either;

public class TestSuite {

	public static class XOR {
		private static Mat[][] dataset() {
			double[][] __input = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
			Mat[] input = { 
					new Mat(2, 1, CvType.CV_32FC1), 
					new Mat(2, 1, CvType.CV_32FC1),
					new Mat(2, 1, CvType.CV_32FC1), 
					new Mat(2, 1, CvType.CV_32FC1) 
			};
			for (int i = 0; i < input.length; i++) {
				input[i].put(0, 0, __input[i]);
			}

			double[][] __expectedOutput = { { 0 }, { 1 }, { 1 }, { 0 } };
			Mat[] expectedOutput = { 
					new Mat(1, 1, CvType.CV_32FC1), 
					new Mat(1, 1, CvType.CV_32FC1),
					new Mat(1, 1, CvType.CV_32FC1), 
					new Mat(1, 1, CvType.CV_32FC1) 
			};
			for (int i = 0; i < expectedOutput.length; i++) {
				expectedOutput[i].put(0, 0, __expectedOutput[i]);
			}
			return new Mat[][] { input, expectedOutput };
		}
		
//		public static void test() {
//			Mat[][] XOR_dataset = XOR.dataset();
//			Mat[] input = XOR_dataset[0];
//			Mat[] expectedOutput = XOR_dataset[1];
//
//		    Layer[] network = {
//		    		Layer.Dense(new int[]{2, 3}, Activation::TANH),
//		    		Layer.Dense(new int[]{3, 1}, Activation::TANH),
//		    };		    
//			int epochs = 10000;
//			double learning_rate = 0.1;
//		    
//		    for (int e = 0; e < epochs; ++e) {
//		    	double error = 0;
//		    	for (int i = 0; i < input.length; ++i) {
////		    		(forward)
//					Mat output = input[i];
//					for (int j = 0; j < network.length; ++j) {
//						output = network[j].forward_propogate_with(output);
//					}
//
////		    		(error)
//					error += Loss.MeanSquaredError.between(expectedOutput[i], output).val[0];
//
////		    		(backward)
//					Mat gradient = Loss.MeanSquaredError.prime(expectedOutput[i], output);
//
//		    		for (int j = network.length - 1; j >= 0; --j) {
//		    			gradient = network[j].backward_propogate_with(gradient, learning_rate);
//		    		}
//		    	}
//		    	error /= input.length;
//		    	Debug.printc("%d/%d, error=%f", e + 1, epochs, error);
//		    }
//		    
//	Debug.prints("========================================================================");
//		    for (int i = 0; i < input.length; ++i) {
//		    	Mat output = input[i];
//		    	for (int j = 0; j < network.length; ++j) {
//		    		output = network[j].forward_propogate_with(output);
//		    	}
//		    	Debug.prints(output.dump());
//		    	Debug.prints(expectedOutput[i].dump());
//		    	System.out.println();
//		    }
//		    return;
//		}
		
		public static void test(Object ...args) {
			Mat[][] XOR_dataset = XOR.dataset();
			Mat[] input = XOR_dataset[0];
			Mat[] expectedOutput = XOR_dataset[1];

		    Layer[] network = {
		    		Layer.Dense(new int[]{2, 3}, Activation::TANH),
		    		Layer.Dense(new int[]{3, 1}, Activation::TANH),
		    };		    
		    Model.instantiate(network, null);
		    network = Model.network;
		    
			int epochs = 10000;
			double learning_rate = 0.1;
		    
		    for (int e = 0; e < epochs; ++e) {
		    	double error = 0;
		    	for (int i = 0; i < input.length; ++i) {
//		    		(forward)
					Mat output = input[i];
					for (int j = 0; j < network.length; ++j) {
						output = network[j].forward_propogate_with(output);
					}

//		    		(error)
					error += Loss.MeanSquaredError.between(expectedOutput[i], output).val[0];

//		    		(backward)
					Mat gradient = Loss.MeanSquaredError.prime(expectedOutput[i], output);

		    		for (int j = network.length - 1; j >= 0; --j) {
		    			gradient = network[j].backward_propogate_with(gradient, learning_rate);
		    		}
//if(args.length == 0)break;
		    	}
		    	error /= input.length;
		    	Debug.printc("%d/%d, error=%f", e + 1, epochs, error);
//if(args.length == 0)break;
		    }
		    
	Debug.prints("========================================================================");
		    for (int i = 0; i < input.length; ++i) {
		    	Mat output = input[i];
		    	for (int j = 0; j < network.length; ++j) {
		    		output = network[j].forward_propogate_with(output);
		    	}
		    	Debug.prints(output.dump());
		    	Debug.prints(expectedOutput[i].dump());
		    	System.out.println();
		    }
		    return;
		}
		
		public static void revised_test() throws InterruptedException {
			Layer[] network = {
		    		Layer.Dense(new int[]{2, 3}, Activation::TANH),
//		    		Activation.Sigmoid(),
//		    		Layer.Dense(new int[]{3, 1}, Activation::TANH),
		    		Layer.Dense(new int[]{3, 1}, null),
//		    		Activation.SoftMax(),
//		    		Activation.TANH()
//		    		Activation.Sigmoid(),
		    		Activation.RELU()
		    };
			Model model = Model.instantiate(network, Either.create_with(List.of(-1, 1)));
//			model.compile_with(Optimizer.GradientDescent, Loss.BinaryCrossEntropyLoss.class);
			model.compile_with(Optimizer.StochasticGradientDescent, Loss.MeanSquaredError.class);
//			model.set_learning_rate(1);
	  
			Either<Random, Integer> state = Either.create_with(42);
			Mat[][] XOR_dataset = XOR.dataset();
			Mat[] input = XOR_dataset[0];
			Mat[] expectedOutput = XOR_dataset[1];
			
			ArrayList<Mat> normalized = Arrays.stream(input).collect(Collectors.toCollection(ArrayList::new));
//			ArrayList<Mat>[] allocated = ModelUtils.allocate(normalized, 0.60, state, true, null);
//			Mat[] fortraining = allocated[0].parallelStream().toArray(Mat[]::new);
			Mat[] fortraining = input;
			  
//			allocated = ModelUtils.allocate(allocated[1], 0.5, state, true, null);
			Object _displable = model.fit(fortraining, /*batch_size=*/ 25, /*epochs=*/ 1000, /*verbose=*/ 1, /*validation=*/ expectedOutput, /*shuffle=*/ true);

//			Debug.print("========================================");
//			model.display(Model.TOTAL_ERROR | Model.COLLAPSE);
			model.display(Model.COLLAPSE);
//			Debug.print("========================================");
			
			Mat prediction = model.predict(expectedOutput[0]);
			  
//			model.save();

		}

	}
	
	public void allocation() {
        ArrayList<Integer> labels = new ArrayList<>();
        labels.add(2);
        labels.add(2);
        labels.add(2);
        labels.add(3);
        labels.add(3);
        labels.add(3);
        labels.add(4);
        labels.add(4);
        labels.add(4);
        ArrayList<Integer> list = IntStream.rangeClosed(1, 9).boxed().collect(Collectors.toCollection(ArrayList::new));
        Debug.printo(list);
        ArrayList<?>[] result = ModelUtils.allocate(list, 0.75, ModelUtils.Either.create_with(12), true, labels);
        System.out.println(Arrays.toString(result));
        
        Debug.printo(list);
        list = IntStream.rangeClosed(1, 9).boxed().collect(Collectors.toCollection(ArrayList::new));
        result = ModelUtils.allocate(list, 0.75, ModelUtils.Either.create_with(12), true, null);
        System.out.println(Arrays.toString(result));
        
        Debug.printo(list);
        list = IntStream.rangeClosed(1, 9).boxed().collect(Collectors.toCollection(ArrayList::new));
        result = ModelUtils.allocate(list, 0.75, ModelUtils.Either.create_with(12), false, null);
        System.out.println(Arrays.toString(result));

	}
	
	public static void printNamed() {
		Mat input = new Mat(3, 3, CvType.CV_32FC3);
		double[] inputData = {
				1, 6, 2,
				5, 3, 1,
				7, 0, 4,
				
				1, 6, 2,
				5, 3, 1,
				7, 0, 4,
				
				1, 6, 2,
				5, 3, 1,
				7, 0, 4,
		};
		input.put(0, 0, inputData);
		
		int args = 0;
		
		Debug.printNamed(new Runnable() {
			@Override
			public void run() {
				Debug.print(args, input);
			}
		}, input);

		Debug.printNamed(new Runnable() {
			@Override
			public void run() {
				Debug.print(args, input);
			}
		}, new Runnable() {
			@Override
			public void run() {
				Debug.print(args, input);
			}
		}, input);

		Debug.printNamed(new Runnable() {
			@Override
			public void run() {
				Debug.print(args, input);
			}
		}, new Runnable() {
			@Override
			public void run() {
				Debug.print(args, input);
			}
		}, input, input);

		Debug.printNamed(input, input, input);

	}

//	public static class ErrorRate extends GUI.DynamicGraph {
//		public static Double __increment() {
//			return null;
//		}
//	}
	
//	private class Tuple
	
	private class ConvolutionalLayer {
		public static void correlations() {
			Mat input = new Mat(3, 3, CvType.CV_32F);
			double[] inputData = {
					1, 6, 2,
					5, 3, 1,
					7, 0, 4
			};
			input.put(0, 0, inputData);
			Debug.print(input);
			
			Mat kernel = new Mat(2, 2, CvType.CV_32F);
			double[] kernelData = {
					1, 2,
					-1, 0
			};
			kernel.put(0, 0, kernelData);
			Debug.print(kernel);
			
//			NOTE: ['valid' Cross Correlation]
			Mat validCrossCorrelation = new Mat();	
			Imgproc.matchTemplate(input, kernel, validCrossCorrelation, Imgproc.TM_CCORR); // valid cross correlation
			Debug.printSurrounded("validCrossCorrelation", validCrossCorrelation);

//			NOTE: ['full' Cross Correlation]
			Mat leftBottomPaddedInput = new Mat();
			Core.copyMakeBorder(input, leftBottomPaddedInput, 0, 1, 0, 1, Core.BORDER_CONSTANT, Scalar.all(0));
			Mat leftBottomPaddedKernel = new Mat();
	        Core.copyMakeBorder(kernel, leftBottomPaddedKernel, 0, 1, 0, 1, Core.BORDER_CONSTANT, Scalar.all(0));
	        Mat fullCrossCorrelation = new Mat();
	        Imgproc.filter2D(leftBottomPaddedInput, fullCrossCorrelation, CvType.CV_32F, leftBottomPaddedKernel, new Point(-1, -1), 0, Core.BORDER_CONSTANT);
			Debug.printSurrounded("fullCrossCorrelation", fullCrossCorrelation);

		}
		
		public static void validCrossCorrelation(Mat input, Mat kernel) {
			Mat validCrossCorrelation = new Mat();	
			Imgproc.matchTemplate(input, kernel, validCrossCorrelation, Imgproc.TM_CCORR); // valid cross correlation
			Debug.printSurrounded("validCrossCorrelation", validCrossCorrelation);			
		}
		
		public static void fullCrossCorrelation(Mat input, Mat kernel) {
			Mat leftBottomPaddedInput = new Mat();
			Core.copyMakeBorder(input, leftBottomPaddedInput, 0, 1, 0, 1, Core.BORDER_CONSTANT, Scalar.all(0));
			Mat leftBottomPaddedKernel = new Mat();
	        Core.copyMakeBorder(kernel, leftBottomPaddedKernel, 0, 1, 0, 1, Core.BORDER_CONSTANT, Scalar.all(0));
	        Mat fullCrossCorrelation = new Mat();
	        Imgproc.filter2D(leftBottomPaddedInput, fullCrossCorrelation, CvType.CV_32F, leftBottomPaddedKernel, new Point(-1, -1), 0, Core.BORDER_CONSTANT);
			Debug.printSurrounded("fullCrossCorrelation", fullCrossCorrelation);
		}
	}
	
	public static void main(String[] args) throws InterruptedException, IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//		TODO: implement implicit shape management {https://stackoverflow.com/questions/54423078/how-are-the-output-size-of-maxpooling2d-conv2d-upsampling2d-layers-calculated}
//		https://www.kaggle.com/code/manjilkarki/realcat-identifier/notebook

//		ArrayList<BufferedImage> normalized = new ArrayList<BufferedImage>();
//
////      TODO: make Main.collect images take Stream<Path> so all this work will be done inside
//		try (Stream<Path> filepaths = Files.walk(Paths.get("training_media/archive"))) {
//			ArrayList<String> filenames = filepaths
//					.collect(Collector.of(ArrayList<String>::new, (arrayList, element) -> {
//						if (element.toFile().isFile()) {
//							arrayList.add(element.subpath(1, element.getNameCount()).toString());
//						}
//					}, (left, right) -> {
//						left.addAll(right);
//						return left;
//					}));
//			Path training_media = Paths.get("training_media");
//			ArrayList<Path> images = Main.collectImages(filenames, training_media);
//			normalized = Main.normalizeImages(images);
//		} catch (Exception exception) {
//			exception.printStackTrace();
//		}
//
//		if (normalized.isEmpty()) {
//			throw new InterruptedException("Failed to normalize BufferedImages");
//		}
//
//		System.out.println(normalized.size());
//
//		Layer[] layers = {
//				Layer.Convolution2D(new int[]{256, 256, 32}, 3, 3, Activation::RELU),
//				Layer.dropout(0.5, null),
//				Layer.MaxPooling2D(new int[]{51, 51, 32}, 3, 5 /*should be default*/, 5, null),
//				Layer.Convolution2D(new int[]{51, 51, 64}, 3, 3, Activation::RELU),
//				Layer.dropout(0.7, null),
//				Layer.MaxPooling2D(new int[]{51, 51, 64}, 3, 5, 5, null),
//				Layer.Flatten(),
//				Layer.Dense(new int[]{1, 128}, Activation::Sigmoid),
//				Layer.dropout(0.7, null),
//				Layer.Dense(new int[]{1, 32}, Activation::RELU),
//				Layer.dropout(0.7, null),
//				Layer.Dense(new int[]{1, 2}, Activation::SoftMax)
//		};
		
		Debug.printSurrounded("TestSuite", null);
		String pathname = "C:\\Users\\mysti\\Documents\\csc 142\\java\\honors-module\\Cat-Finder\\training_media\\archive\\cats\\CAT_01\\00000184_022.jpg";
////		String pathname = "C:\\Users\\mysti\\Documents\\csc 142\\java\\honors-module\\Cat-Finder\\training_media\\archive\\cats\\CAT_01\\00000184_002.jpg";
		BufferedImage image = null;
		try {
			image = ImageIO.read(new File(pathname));
		} catch (IOException exception) { exception.printStackTrace();}
//		
		Mat matrix = Main.toMatrix(image);
//		Imgproc.cvtColor(matrix, matrix, Imgproc.COLOR_BGR2RGB);
		Imgproc.cvtColor(matrix, matrix, Imgproc.COLOR_BGR2GRAY);
//		Imgproc.resize(matrix, matrix, new Size(256, 256)); // NOTE: increase because actual is -2
//		
		MatOfByte buffer = new MatOfByte();
		Imgcodecs.imencode(".jpg", matrix, buffer);
		InputStream stream = new ByteArrayInputStream(buffer.toArray());
		image = ImageIO.read(stream);
		
//		Main.display(image);
		
		
		
//		ConvolutionalLayer.correlations();
//		Convolution2D conv = Layer.Convolution2D(2, new int[] {2, 2, 3}, new int[] {3, 3, 3}, 3, null);
//		Mat input = new Mat(3, 3, CvType.CV_32FC3);
//		double[] inputData = {
//				1, 6, 2,
//				5, 3, 1,
//				7, 0, 4,
//				
//				1, 6, 2,
//				5, 3, 1,
//				7, 0, 4,
//				
//				1, 6, 2,
//				5, 3, 1,
//				7, 0, 4,
//		};
//		input.put(0, 0, inputData);
//		Debug.print(input);
//		
////		NOTE: Expected result Mat<2, 2>.{CvType.CV_32FC3}
//		Mat next = conv.forward_propogate_with(input);
//		
//		Debug.print(next);
//		
//		Debug.printSurrounded("null", null);
//		Mat gradient = conv.backward_propogate_with(next, 1);
//		
//		Debug.printSurrounded("Gradient", null);
//		Debug.printNamed(gradient);
//		
		
//		kernelS
//		Layer.Convolution2D(new int[]{256, 256, 32}, 3, 3, Activation::RELU);
		

		
//      Main.display_with(image, null, pathname);
//		Main.display(image);

//		matrix.convertTo(matrix, CvType.CV_32F + (8 * (matrix.channels() - 1)));
//		Core.divide(matrix, new Scalar(255), buffer);
//		Layer[] network = new Layer[] {
//				Layer.Convolution2D(32, new int[] {3, 3}, new int[]{matrix.channels(), matrix.rows(), matrix.cols()}, matrix.channels(), Activation::RELU),
//				
//		};
//		Mat images = Dnn.blobFromImages(List.of(matrix), 1.0);
//		Debug.print(images.toString());
//		Debug.print(new int[]{images.channels(), images.rows(), images.cols(), images.depth()});
//		Debug.print(new int[]{images.size(0), images.size(1), images.size(2), images.size(3)});
//		
//		Debug.printSurrounded("FORWARD", new int[]{matrix.channels(), matrix.rows(), matrix.cols()});
////		Mat output = network[0].forward_propogate_with(matrix);
//		Mat output = network[0].forward_propogate_with(images);
		
//		https://chat.openai.com/c/3bb73d24-e171-4fb2-8037-d1a6def8c031
		Debug.print(true);
		Debug.printSurrounded("TestSuite", null);
		
//		input.release();
//		next.release();
	}
}


