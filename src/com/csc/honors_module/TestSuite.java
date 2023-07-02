package com.csc.honors_module;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

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
		    Model.initiate(network, null);
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

	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
//		XOR.test();
		Layer[] network = {
	    		Layer.Dense(new int[]{2, 3}, Activation::TANH),
//	    		Activation.Sigmoid(),
	    		Layer.Dense(new int[]{3, 1}, Activation::TANH),
//	    		Activation.Sigmoid(),
	    };
		Model model = Model.initiate(network, Either.create_with(List.of(-1, 1)));
//		model.compile_with(Optimizer.GradientDescent, Loss.BinaryCrossEntropyLoss.class);
		model.compile_with(Optimizer.GradientDescent, Loss.MeanSquaredError.class);
  
		Either<Random, Integer> state = Either.create_with(42);
		Mat[][] XOR_dataset = XOR.dataset();
		Mat[] input = XOR_dataset[0];
		Mat[] expectedOutput = XOR_dataset[1];
		
		ArrayList<Mat> normalized = Arrays.stream(input).collect(Collectors.toCollection(ArrayList::new));
//		ArrayList<Mat>[] allocated = ModelUtils.allocate(normalized, 0.60, state, true, null);
//		Mat[] fortraining = allocated[0].parallelStream().toArray(Mat[]::new);
		Mat[] fortraining = input;
		  
//		allocated = ModelUtils.allocate(allocated[1], 0.5, state, true, null);
		Object displable = model.fit(fortraining, /*batch_size=*/ 25, /*epochs=*/ 1000, /*verbose=*/ 1, /*validation=*/ expectedOutput, /*shuffle=*/ true);
//				TODO: display displayable to show progress
		  
		
//		BufferedImage image = ImageIO.read(new File(pathname));
//		Position<Float, Float> 
		Object prediction = model.predict(expectedOutput[0]);
//		Debug.printo(prediction);
//		Main.display(image);
		  
//		model.save();
	}
}
