/**
 * 
 */
package com.csc.honors_module;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author mysti
 *
 */
public class MathUtils {
//	https://stackoverflow.com/questions/11032781/fastest-way-to-generate-binomial-coefficients
	public static class BinomialDistribution {
//		NOTE: memoize
		public static long binomial_coefficient(int n, int k) {
			if ((n == k) || (k == 0))
				return 1;
			return binomial_coefficient(n - 1, k) + binomial_coefficient(n - 1, k - 1);
		}
		
//		NOTE: memoize
		public static BigInteger choose(final int N, final int K) {
			BigInteger result = BigInteger.ONE;
			for (int k = 0; k < K; ++k) {
				result = result.multiply(BigInteger.valueOf(N - k)).divide(BigInteger.valueOf(k + 1));
			}
			return result;
		}
	}
	
	public static class Random extends java.util.Random {
		/**
		 * 
		 */
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
//		    """Computes approximate mode of multivariate hypergeometric.
//
//		    This is an approximation to the mode of the multivariate
//		    hypergeometric given by class_counts and n_draws.
//		    It shouldn't be off by more than one.
//
//		    It is the mostly likely outcome of drawing n_draws many
//		    samples from the population given by class_counts.
//
//		    Parameters
//		    ----------
//		    class_counts : ndarray of int
//		        Population per class.
//		    n_draws : int
//		        Number of draws (samples to draw) from the overall population.
//		    rng : random state
//		        Used to break ties.
//
//		    Returns
//		    -------
//		    sampled_classes : ndarray of int
//		        Number of samples drawn from each class.
//		        np.sum(sampled_classes) == n_draws
//			"""

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
