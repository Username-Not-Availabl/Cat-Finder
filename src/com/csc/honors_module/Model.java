package com.csc.honors_module;

import java.awt.image.BufferedImage;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import com.csc.honors_module.MathUtils.Activation;
import com.csc.honors_module.MathUtils.Loss;
import com.csc.honors_module.ModelUtils.Either;
import com.csc.honors_module.ModelUtils.Position;

public class Model implements Serializable {
	private static final long serialVersionUID = 3191010016490991315L;

//	https://www.linkedin.com/pulse/opencv-java-yolo-object-detection-images-svetozar-radoj%C4%8Din
//	https://github.com/mesutpiskin/opencv-object-detection/blob/master/src/DeepNeuralNetwork/DnnProcessor.java
	
//	https://docs.djl.ai/jupyter/tutorial/02_train_your_first_model.html
//	https://towardsdatascience.com/deep-learning-in-java-d9b54ae1423a
	
//	https://github.com/yacineMahdid/artificial-intelligence-and-machine-learning/blob/master/Neural%20Network%20from%20Scratch%20in%20Java/src/Layer.java
//	https://www.youtube.com/watch?v=1DIu7D98dGo
	
//	https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
//	https://www.cs.swarthmore.edu/~meeden/cs81/s10/BackPropDeriv.pdf
	
	public static class Data implements Serializable {
		double[] data;
		double[] expected_output;
		
		public Data(double[] training_data, double[] expected_output) {
			this.data = training_data;
			this.expected_output = expected_output;
		}
	}
	
	public static Layer[] network;
	public static Mat[] training_data_set;
	
	private int EPOCHS;
	private double learning_rate;
	
	private Optimizer optimizer;
	private Class<? extends Loss> loss;
	private Mat[] validation_data_set;
	
//	public static enum Activation {
//		ReLu, Sigmoid, TANH;
//	}
	
	public static Model initiate(Layer[] network, Either<Range<Double>, ArrayList<Double>> range) {
		Neuron.setWeightRange(Either.create_with(List.of(-1, 1)));
		
		ArrayList<Layer> layers = new ArrayList<>();
//		layers.add(Layer.Input());
		for (Layer layer: network) {
			layers.add(layer);
			Supplier<Activation> function;
			try {
				if ((function = ((CanHaveActivationFunction)layer).activation()) != Debug.NULL()) {
					layers.add(function.get());
				}
			} catch (ClassCastException exception) {}
		}
		
//		layers.stream().forEach(System.out::println);
		Model.network = layers.toArray(new Layer[0]);
		
		
		
//	 	NOTE: this is probably where you would use the training data
		
//		NOTE: should probably be toggleable
//		Debug.prints("======================");
//		Debug.prints("=======BEFORE=========");
//		Debug.prints("======================");
//		for (Data element : training_data_set) {
//			Model.forward_propogate(element.data);
//			Debug.print(layers[layers.length - 1].neurons[0].value);
//		}
		
//		TODO: create instance, affect instance 
//		and then return it
		return new Model();
	}
	
	private static void train(int iterations, float learning_rate) {
		for (int i = 0; i < iterations; ++i) {
//			for (int j = 0; j < )
//			https://github.com/yacineMahdid/artificial-intelligence-and-machine-learning/blob/master/Neural%20Network%20from%20Scratch%20in%20Java/src/NeuralNetwork.java#L66
		}
	}
	
//	Stochastic gradient descent
	public static enum Optimizer { GradientDescent, Adam; }
	public <T extends Loss> Model compile_with(Optimizer optimizer, Class<T> loss) throws InvalidParameterException {
		if (optimizer == Optimizer.Adam) {
			throw new InvalidParameterException("!(unimplemented): Adaptive moment estimation is not implemented yet!");
		}
		
		this.optimizer = optimizer;
		this.loss = loss;
//		train(1000000, 0.05f);
//		
//		Debug.prints("======================");
//		Debug.prints("========AFTER=========");
//		Debug.prints("======================");
//		for (Data element : training_data_set) {
//			Model.forward_propogate(element.data);
//			Debug.print(layers[layers.length - 1].neurons[0].value);
//		}
		return null;
	}
	
//	public static void forward_propogate(double[] inputs) {
////		for (int i = 0; i < layers.length; ++i) {
////			for (int j = 0; j < layers[i].neurons.length; ++j) {
////				double sum = 0;
////				for (int k = 0; k < layers[i - 1].neurons.length; ++k) {
////					sum += layers[i - 1].neurons[k].value * layer[i].neurons[j].weights[k];
////				}
//////				sum += layers[i].neurons[j].bias;
////				layers[i].neurons[j].value = MathUtils.Sigmoid(sum);
////			}
////		}
//	}
	
	private double loss(Mat actual, Mat predicted) throws NoSuchMethodException, SecurityException, IllegalAccessException, InvocationTargetException {
		Method between = this.loss.getMethod("between", Mat.class, Mat.class);
		return ((Scalar)between.invoke(null, actual, predicted)).val[0];
	}
	
	private Mat gradient(Mat actual, Mat predicted) throws NoSuchMethodException, SecurityException, IllegalAccessException, InvocationTargetException {
		Method between = this.loss.getMethod("prime", Mat.class, Mat.class);
		return ((Mat)between.invoke(null, actual, predicted));
	}
	
//	Object displable = model.fit(fortraining, /*batch_size=*/ 25, /*epochs=*/ 25, /*verbose=*/ 1, /*validation=*/ allocated[1], /*shuffle=*/ true);
	public Model fit(Mat[] training_data, int batch_size, int epochs, int verbose, Mat[] validation_data, boolean shuffle) {
	    this.EPOCHS = epochs;
	    this.training_data_set = training_data;
	    this.validation_data_set = validation_data;
	    this.learning_rate = 0.1;
	    
int FAIL = 11;
	    
//	    TODO: implement use of batch_size
//	    You randomly collect a batch of size batch_size samples
//	    and use that for gradient descent instead of using the total average.
//	    Less accurate but enhances performance
	    
		for (int e = 0; e < epochs; ++e) {
	    	double error = 0;
	    	for (int i = 0; i < training_data.length; ++i) {
//	    		(forward)
				Mat output = training_data[i];
				for (int j = 0; j < network.length; ++j) {
//					Debug.print(network[j], output.dump());
					output = network[j].forward_propogate_with(output);
				}
//if (e == 9) break;
//	    		(error)
				try {
					error += this.loss(validation_data[i], output);
//					Debug.print(error);
				} catch (NoSuchMethodException exception) {
					exception.printStackTrace();
					error += Loss.MeanSquaredError.between(validation_data[i], output).val[0];
				} catch (SecurityException | IllegalAccessException | InvocationTargetException exception) {}
//				error += Loss.BinaryCrossEntropyLoss.between(validation_data[i], output).val[0];


//if (e == FAIL) {
//	Debug.print(error); 
//	Debug.print(validation_data[i + 1].dump()); 
//	Debug.print(output.dump());
//	try {
//		double a = this.loss(validation_data[i], output);
//		Debug.printSurrounded("a", a);
//	} catch (NoSuchMethodException | SecurityException | IllegalAccessException | InvocationTargetException e1) {
//		// TODO Auto-generated catch block
//		e1.printStackTrace();
//	}
//	break;
//}
if (Double.isNaN(error)) {
	Debug.print(validation_data[i].dump()); 
	Debug.print(output.dump());
	try {
		double a = this.loss(validation_data[i], output);
		Debug.printSurrounded("a", a);
	} catch (NoSuchMethodException | SecurityException | IllegalAccessException | InvocationTargetException e1) {
		// TODO Auto-generated catch block
		e1.printStackTrace();
	}
	FAIL = e;
	break;
}
//	    		(backward propogation)
				Mat gradient = new Mat();
				switch (this.optimizer) {
					case GradientDescent: {
						try {
							gradient = this.gradient(validation_data[i], output);
						} catch (NoSuchMethodException exception) {
							exception.printStackTrace();
							gradient = Loss.MeanSquaredError.prime(validation_data[i], output);
						} catch (Exception exception) {};
						break;
					}
					case Adam: {
						Debug.print("!(unimplemented): Adaptive moment estimation is not implemented yet!");
						break;
					}
				}
//				gradient = Loss.BinaryCrossEntropyLoss.prime(validation_data[i], output);
//if (e == 0) break;
	    		for (int j = network.length - 1; j >= 0; --j) {
	    			gradient = network[j].backward_propogate_with(gradient, learning_rate);
	    		}
	    	}
//if (e == FAIL) break;
	    	error /= training_data.length;
	    	Debug.printc("%d/%d, error=%f", e + 1, epochs, error);
	    }
		return null;
	}
	
//	public Position<Float, Float> predict(BufferedImage image) {
//		for (int i = 0; i < this.training_data_set.length; ++i) {
//	    	Mat output = this.training_data_set[i];
//	    	for (int j = 0; j < network.length; ++j) {
//	    		output = network[j].forward_propogate_with(output);
//	    	}
//	    	Debug.prints(output.dump());
//	    	Debug.prints(this.validation_data_set[i].dump());
//	    	System.out.println();
//	    }
//		return null;
//	}

	public Object predict(Mat input) {
		System.out.println("=================================================");
		for (int i = 0; i < this.training_data_set.length; ++i) {
	    	Mat output = this.training_data_set[i];
	    	for (int j = 0; j < network.length; ++j) {
	    		output = network[j].forward_propogate_with(output);
	    	}
	    	Debug.prints(output.dump());
	    	Debug.prints(this.validation_data_set[i].dump());
	    	System.out.println();
	    }
		return null;
	}

	synchronized public void save(String ...filepaths) throws FileNotFoundException, IOException {
//		TODO: possibly traverse the path and save by number
		for (int i = 0; i < filepaths.length; ++i) {
			try (ObjectOutputStream stream = new ObjectOutputStream(new FileOutputStream(filepaths[i]))) {
				stream.writeObject(this);
			}
		}
	}

}
