/**
 * 
 */
package com.csc.honors_module;

import java.math.BigInteger;
import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

/**
 * @author mysti
 *
 */
public class MathUtils {
////	https://stackoverflow.com/questions/11032781/fastest-way-to-generate-binomial-coefficients
//	public static class BinomialDistribution {
////		NOTE: memoize
//		public static long binomial_coefficient(int n, int k) {
//			if ((n == k) || (k == 0))
//				return 1;
//			return binomial_coefficient(n - 1, k) + binomial_coefficient(n - 1, k - 1);
//		}
//		
////		NOTE: memoize
//		public static BigInteger choose(final int N, final int K) {
//			BigInteger result = BigInteger.ONE;
//			for (int k = 0; k < K; ++k) {
//				result = result.multiply(BigInteger.valueOf(N - k)).divide(BigInteger.valueOf(k + 1));
//			}
//			return result;
//		}
//	}
	
	public static class Random extends java.util.Random {
		private static final long serialVersionUID = -5756005556625431208L;

		@SuppressWarnings("unchecked")
		public static <T extends Object> T[] choice(T[] list, int size, Boolean with_replacement, java.util.Random random) {
//			https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/utils/__init__.py#L1138
			T[] result = (T[]) new Object[size];
			for (int i = 0; i < size; ++i) {
				T value = list[random.nextInt(list.length)];
				if (!with_replacement) {
					if (Arrays.stream(result).anyMatch(element -> element == value)) {
						--i;
						continue;
					}
				}
				result[i] = value;
			}
			return result;
		}
		
		public static double between(Range<Double> range) {
			return ThreadLocalRandom.current().nextDouble(range.from.get(), range.to.get());
		}
	}
	
	public static double extend_by_predicate(Boolean predicate, double factor, double value) {
		if (predicate) {
			return factor * value;
		} else 
			return value;
	}
	
	public abstract static class Activation extends Layer {

		public Activation() {};
		private Mat input;
		
		public Mat forward_propogate_with(Mat input) {
			this.input = input;
			return activation(this.input);
		}
		
		public Mat backward_propogate_with(Mat output_gradient, double learning_rate) {
			return output_gradient.mul(activation_derivative(this.input));
		}
		
		protected abstract Mat activation(Mat input);
		protected abstract Mat activation_derivative(Mat input);

//		https://en.wikipedia.org/wiki/Sigmoid_function
		public static class Sigmoid  extends Activation {
			public static double logistic(double x, double supremum, double growthrate, double midpoint_x) {
				double exponent = -growthrate * (x - midpoint_x);
				return supremum / (1 + Math.pow(Math.E, exponent));
			}
			
			public static double expit(double x) {
				return logistic(x, 1, 1, 0);
			}

			public static double expit_prime(double x) {
				double s = expit(-x);
				return s * (1 - s);
			}
			
			protected Mat activation(Mat input) {
				Mat clone = input.clone();
				Matrix.create_from(clone)
					  .for_each((x, y) -> {
					Matrix.replace_at(clone, new int[]{x, y}, (a) -> expit(a));
				});
				return clone;
			}
			
			protected Mat activation_derivative(Mat input) {
				Mat clone = input.clone();
				Matrix.create_from(clone)
					  .for_each((x, y) -> {
					Matrix.replace_at(clone, new int[]{x, y}, (a) -> expit_prime(a));
				});
				return clone;
			}
		}
		public static Sigmoid Sigmoid() { return new Sigmoid(); }
		
		public static class TANH extends Activation {
			private static double tanh_prime (double x) {
				return 1 - Math.pow(Math.tanh(x), 2);
			}
			
			protected Mat activation(Mat input) {
				Mat clone = input.clone();
				Matrix.create_from(clone)
					  .for_each((x, y) -> {
						  Matrix.replace_at(clone, new int[]{x, y}, (a) -> Math.tanh(a));
					  });
				return clone;
			}
			
			protected Mat activation_derivative(Mat input) {
				Mat clone = input.clone();
				Matrix.create_from(clone)
					  .for_each((x, y) -> {
						  Matrix.replace_at(clone, new int[]{x, y}, (a) -> tanh_prime(a));
					  });
				return clone;
			}
		}
		public static TANH TANH() { return new TANH(); }
		
		public static class ReLu extends Activation {
			protected Mat activation(Mat input) {
				Mat clone = input.clone();
				Matrix.create_from(clone).for_each((x, y) -> {
					    Matrix.replace_at(clone, new int[]{x, y}, (a) -> Math.max(0, a));
				});
				return clone;
			}
			
			protected Mat activation_derivative(Mat input) {
				Mat clone = input.clone();
				Matrix.create_from(clone).for_each((x, y) -> {
					Matrix.replace_at(clone, new int[]{x, y}, (a) -> {
						if (a > 0) return 1.0;
						if (a < 0) return 0.0;
						return null;
					});
				});
				return clone;
			}
		}
		public static ReLu RELU() { return new ReLu(); }
	}
	
	public static class Loss {
//		public static Scalar between(Mat actual, Mat predicted) {return null;}
//		public static Mat prime(Mat actual, Mat predicted) {return null;}
		
		public static class MeanSquaredError extends Loss {
			public static Scalar between(Mat actual, Mat predicted) {
				Mat destination = new Mat();
				Core.absdiff(actual, predicted, destination);
				Core.multiply(destination, destination, destination);
				return Core.mean(destination);
			}
			
			public static Mat prime(Mat actual, Mat predicted) {
				Mat destination = new Mat();
				Core.subtract(predicted, actual, destination);
				Matrix.create_from(destination)
					  .for_each((x, y) -> {
					Matrix.replace_at(destination, new int[]{x, y}, (element) -> {
						return (2 / (double)destination.rows()) * element;
					});
				});
				return destination;
			}
		}
		
		public static class BinaryCrossEntropyLoss extends Loss {
			public static Scalar between(Mat actual, Mat predicted) {
				Mat lhs = predicted.clone();
				Matrix.create_from(lhs).for_each((x, y) -> {
					Matrix.replace_at(lhs, new int[]{x, y}, (element) -> {
						if (element < 0) {
							throw new InvalidParameterException("Mat::predicted cannot have negative inputs");
						}
						return Math.log(element);
					});
				});
				Core.multiply(actual, lhs, lhs, -1);
//				Debug.print(lhs);
				
				Mat rhs = predicted.clone();
				Matrix.create_from(rhs).for_each((x, y) -> {
					Matrix.replace_at(rhs, new int[] {x, y}, (element) -> {
						return Math.log(1 - element);
					});
				});
				Mat inverse = actual.clone();
				Matrix.create_from(inverse).for_each((x, y) -> {
					Matrix.replace_at(inverse, new int[] {x, y}, (element) -> {
						return (1 - element);
					});
				});
				Core.multiply(inverse, rhs, rhs);
				
				Mat result = new Mat();
				Core.subtract(lhs, rhs, result);
				return Core.mean(result);
			}

			public static Mat prime(Mat actual, Mat predicted) {
				Mat __actual = predicted.clone();
				Matrix.transform_using(__actual, (element) -> (1 - element));

				Mat __predicted = predicted.clone();
				Matrix.transform_using(__predicted, (element) -> (1 - element));

				Mat destination = new Mat();
				Core.divide(__actual, __predicted, destination);

				Mat division = new Mat();
				Core.divide(actual, predicted, division);

				Mat lhs = new Mat();
				Core.subtract(destination, division, lhs);

				Matrix.create_from(lhs).for_each((x, y) -> {
					Matrix.replace_at(lhs, new int[] { x, y }, (element) -> {
						return (1 / (double) lhs.rows()) * element;
					});
				});
				return lhs;
			}
		}
	}

	public static class MultivariateHypergeometricDistribution {
		public static int[] mode(int[] population, int[] samples) {
			int[] mode = new int[population.length];
			BigInteger maximum = BigInteger.ZERO;

			int[] indices = new int[population.length];
			while (true) {
				BigInteger current_combination_probability = BigInteger.ONE;
				for (int i = 0; i < population.length; ++i) {
					current_combination_probability = current_combination_probability.multiply(choose(population[i], samples[i], indices[i]));
				}
				
				if (current_combination_probability.compareTo(maximum) > 0) {
					maximum = current_combination_probability;
					System.arraycopy(indices, 0, mode, 0, population.length);
				}
				
				int dimension = population.length - 1;
				while ((dimension >= 0) && (indices[dimension] == samples[dimension])) {
					indices[dimension] = 0;
					--dimension;
				}
				
				if (dimension < 0) { break; }
				indices[dimension]++;
			}
			return mode;
		}

//		https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/utils/__init__.py#L1077
		@SuppressWarnings("unchecked")
		public static int[] approximate_mode(int[] population, int samplessize, java.util.Random rng) {

//			NOTE: floored means we don't overshoot n_samples, but probably under shoot
			double sum = Arrays.stream(population).sum();			
			double[] continuous = Arrays.stream(population).mapToDouble(i -> (double)i / sum * (double)samplessize).toArray();
			int[] floored = Arrays.stream(continuous).mapToInt(i -> (int)Math.floor(i)).toArray();
			
//		    NOTE: we add samples according to how much "left over" probability
//		    	  they had, until we arrive at n_samples
			int to_be_added = (int)(samplessize - Arrays.stream(floored).sum());
			if (to_be_added > 0) {
				double[] remainder = IntStream.range(0, floored.length).mapToDouble(i -> continuous[i] - floored[i]).toArray();
				ArrayList<Double> values = (ArrayList<Double>) ModelUtils.unique(Arrays.stream(remainder).boxed().collect(Collectors.toCollection(ArrayList::new)), false)[0];
//		        NOTE: add according to remainder, but break ties
//		        	  randomly to avoid biases
				for (double value: values) {
					int[] passing_indices = IntStream.range(0, remainder.length).filter(i -> remainder[i] == value).toArray();
//		            NOTE: if we need_to_add less than what's in inds
//		            	  we draw randomly from them.
//		            	  if we need to add more, we add them all and
//		            	  go to the next value
					int add = Math.min(passing_indices.length, to_be_added);
//					rng.
//					https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/utils/__init__.py#L1138
					int[] indices = Arrays.stream(Random.choice(Arrays.stream(passing_indices).boxed().toArray(), add, false, rng)).mapToInt(i -> (int)i).toArray();
//					NOTE: behaviour might not be accurate
					floored[indices[0]] += 1;
					to_be_added -= add;
					if (to_be_added == 0) { break; }
				}
			}
			return floored;
		}
		
		public static BigInteger choose(int n, int k, int m) {
			BigInteger numerator = BigInteger.ONE, denominator = BigInteger.ONE;
			
			for (int i = 0; i < k; i++) {
				BigInteger top = BigInteger.valueOf(i < m ? n - i : m);
				BigInteger bottom = BigInteger.valueOf(i < m ? i + 1 : i - (m - 1));
				
				numerator = numerator.multiply(top);
				denominator = denominator.multiply(bottom);
			}
			return numerator.divide(denominator);
		}
	}
	
	public static class MultinomialDistribution {
		
	}
}
