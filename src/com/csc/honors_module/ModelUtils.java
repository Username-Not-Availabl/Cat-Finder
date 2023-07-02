package com.csc.honors_module;

import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;


public class ModelUtils {
//	https://stackoverflow.com/questions/26162407/is-there-an-equivalent-of-scalas-either-in-java-8
	final public static class Either<L extends Object, R extends Object> {
		private Optional<L> left;
		private Optional<R> right;
		
		private enum isempty { LEFT, RIGHT };
		private isempty choice;
		
		private Either(Optional<L> left, Optional<R> right, isempty choice) {
			this.left = left;
			this.right = right;
			this.choice = choice;
		}
		
		@SuppressWarnings("unchecked")
		public static <L extends Object, R extends Object> Either<L, R> create_with(Object initial) {
			Objects.requireNonNull(initial);
			return new Either<L, R>(Optional.ofNullable((L)initial), Optional.empty(), isempty.RIGHT);
		}
		
		@SuppressWarnings("rawtypes")
		private Class acknowledge() {
			switch (this.choice) {
				case LEFT: {
					this.choice = isempty.RIGHT;
					return left.get().getClass();
				}
				case RIGHT: {
					this.choice = isempty.LEFT;
					return right.get().getClass();
				}
				default: throw new RuntimeException("!(Unreachable)");
			}
		}
		
		@SuppressWarnings("unchecked")
		public void become(Object other) {
			Objects.requireNonNull(other);
			if (!acknowledge().isInstance(other))
				throw new IllegalArgumentException("[ERROR]: choice is not of previously declared type");
			
			switch (this.choice) {
				case LEFT: this.right = Optional.ofNullable((R)other);
				case RIGHT: this.left = Optional.ofNullable((L)other);
			}

		}
		
		@SuppressWarnings("unchecked")
		public <T extends Object> T get() {
			switch (this.choice) {
				case LEFT: return (T) this.right.get();
				case RIGHT: return (T) this.left.get();
				default: throw new RuntimeException("!(Unreachable)");
			}
		}
	}
	
	
	public static <T> Random check_random_state(T seed) {
		if (seed instanceof Random) {
			return (Random) seed;
		}
		
		if (seed instanceof Number)
			return new Random(((Number) seed).longValue());
		
		if (!(seed instanceof Number) && !(seed instanceof Random))
			throw new IllegalArgumentException("{%s} cannot be used to seed a [Random] instance".formatted(seed.toString()));
		return null;
	}
	
//	private static void split(Optional<Integer> _splits) {
//		int splits = _splits.isEmpty() ? 10 : _splits.get();
//	}
	
    @SuppressWarnings({ "unchecked", "rawtypes" })
	private static <T extends ArrayList> int[] shape(ArrayList<T> list) {
        ArrayList<Integer> shape = new ArrayList<>();
        shape.add(list.size());

        if (List.class.isInstance(list.get(0))) {
            if (list.stream().anyMatch(i -> i.size() > list.get(0).size())) {
                throw new IllegalArgumentException("Cannot have subLists of inconsistent shape");
            }
            int[] result = shape(list.get(0));
            ArrayList<Integer> asList = IntStream.of(result).boxed().collect(Collectors.toCollection(ArrayList::new));
            shape.addAll(asList);
        }
        
        return shape.stream().mapToInt(Integer::intValue).toArray();
    }

    private static <T> Object create_with_shape(int[] shape) {
        ArrayList<Object> result = new ArrayList<>();
        result.ensureCapacity(shape[0]);

        if (shape.length > 1) {
            for (int i = 0; i < shape[0]; ++i) {
                result.add(create_with_shape(Arrays.copyOfRange(shape, 1, shape.length)));
            }
        } else for (int i = 0; i < shape[0]; ++i) {
            result.add(null);
        }
        
        return result;
    }

    private static void swap(int[] list, int first, int second) {        
        int container = list[first];
        list[first] = list[second];
        list[second] = container;
    }

    private static int[] expandindex(int index, int[] shape) {
        if (shape.length == 1) {
            return new int[] {index % shape[0]};
        }
            
        int[] result = new int[shape.length];
        int product = Arrays.stream(Arrays.copyOfRange(shape, 1, shape.length)).reduce(1, (left, right) -> left * right);

        result[0] = index / product;
        int[] dimensions = expandindex(index, Arrays.copyOfRange(shape, 1, shape.length));
        
        System.arraycopy(dimensions, 0, result, 1, dimensions.length);
        return result;
    }

    @SuppressWarnings("unchecked")
	private static <T extends Number> Integer get(Object _list, int[] coordinates) {
        ArrayList<Object> list = (ArrayList<Object>)_list;

        if (coordinates.length == 1 && !List.class.isInstance(list.get(0))) {
            return (Integer)list.get(coordinates[0]);
        } 

        return (Integer)get((ArrayList<Object>)list.get(coordinates[0]), Arrays.copyOfRange(coordinates, 1, coordinates.length));
    }

    @SuppressWarnings("unchecked")
	private static void populate_with(Object _list, int[] coordinates, int value) {
        ArrayList<Object> list = (ArrayList<Object>)_list;

        if (coordinates.length == 1 && !List.class.isInstance(list.get(0))) {
            if (list.get(coordinates[0]) == null) {
                list.set(coordinates[0], value);
                return;
            }
        } 

        populate_with((ArrayList<Object>)list.get(coordinates[0]), Arrays.copyOfRange(coordinates, 1, coordinates.length), value);
    }
	
//	https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/fromnumeric.py#L550-L594
//	input axes[0, 1]
//	{
//		{1, 2, 3},
//	}
	
//	output
//	{
//	{1}, [(0, 0) -> (0, 0)]
//	{2}, [(0, 1) -> (1, 0)]
//	{3}  [(0, 2) -> (2, 0)]
//	}
	
//	TODO: extend to match functionality
//	input axes[0, 2]
//	{
//		{
//			{0, 1}, 
//			{2, 3}
//		},
//		{
//			{4, 5}, 
//			{6, 7}
//		}
//	}
	
//	output
//	{
//		{
//			{0:[(0, 0, 0) -> (0, 0, 0)], 4:[(1, 0, 0) -> (0, 0, 1)]}, 
//			{2:[(0, 1, 0) -> (0, 1, 0)], 6:[(1, 1, 0) -> (0, 1, 1)]}
//		},
//		{
//			{1:[(0, 0, 1) -> (1, 0, 0)], 5:[(1, 0, 1) -> (1, 0, 1)]}, 
//			{3:[(0, 1, 1) -> (1, 1, 0)], 7:[(1, 1, 1) -> (1, 1, 1)]}
//		},
//	}

    @SuppressWarnings("unchecked")
	private static <T> ArrayList<ArrayList<T>> swapaxes(ArrayList<ArrayList<T>> list, int[] axes) {
        if (axes[0] == axes[1]) {
            throw new IllegalArgumentException("axes must be different.");
        }

        ArrayList<ArrayList<T>> result = new ArrayList<>();
        int[] shape = shape(list);
        swap(shape, axes[0], axes[1]);
        
        result = (ArrayList<ArrayList<T>>)create_with_shape(shape);
        for (int i = 0; i < Arrays.stream(shape).reduce(1, (left, right) -> left * right); ++i) {
            int[] dimensions = expandindex(i, shape);
            int[] reversed = dimensions.clone();
            swap(reversed, axes[0], axes[1]);
            populate_with(result, dimensions, get(list, reversed));
        }

        return result;
    }
	    
    @SuppressWarnings("unchecked")
	public static <T extends Number> ArrayList<Double> cumulative_sum_of(ArrayList<T> list) {
    	Double[] asArray = list.parallelStream().map(Number::doubleValue).toArray(Double[]::new);
    	Arrays.parallelPrefix(asArray, Double::sum);
    	return Arrays.stream(asArray).collect(Collectors.toCollection(ArrayList::new));
    }
    
	public static <T> ArrayList<ArrayList<T>> split_into(ArrayList<T> list, int sections) {
//	    """
//	    Split an array into multiple sub-arrays as views into `ary`.
//
//	    Parameters
//	    ----------
//	    ary : ndarray
//	        Array to be divided into sub-arrays.
//	    indices_or_sections : int or 1-D array
//	        If `indices_or_sections` is an integer, N, the array will be divided
//	        into N equal arrays along `axis`.  If such a split is not possible,
//	        an error is raised.
//
//	        If `indices_or_sections` is a 1-D array of sorted integers, the entries
//	        indicate where along `axis` the array is split.  For example,
//	        ``[2, 3]`` would, for ``axis=0``, result in
//
//	          - ary[:2]
//	          - ary[2:3]
//	          - ary[3:]
//
//	        If an index exceeds the dimension of the array along `axis`,
//	        an empty sub-array is returned correspondingly.
//	    axis : int, optional
//	        The axis along which to split, default is 0.
//
//	    Returns
//	    -------
//	    sub-arrays : list of ndarrays
//	        A list of sub-arrays as views into `ary`.
//		"""
		if (sections <= 0) {
			throw new InvalidParameterException("Number of sections must be > 0");
		}
		
//		.divmod
		int each_section = Math.floorDiv(list.size(), sections);
		int extras = Math.floorMod(list.size(), sections);
		
		ArrayList<Integer> section_sizes = new ArrayList<>();
		section_sizes.add(0);
		section_sizes.addAll(Collections.nCopies(extras, each_section + 1));
		section_sizes.addAll(Collections.nCopies(sections - extras, each_section));
		
		ArrayList<Integer> division_points = cumulative_sum_of(section_sizes).stream().map(i -> Integer.valueOf(i.intValue())).collect(Collectors.toCollection(ArrayList::new));
		
		ArrayList<ArrayList<T>> sub_lists = new ArrayList<>();
		for (int i = 0; i < sections; ++i) {
			int start = division_points.get(i);
			int end = division_points.get(i + 1);
			sub_lists.add(new ArrayList<T>(list.subList(start, end)));
		}
		
		return sub_lists;
	}
	
	public static <T> Object[] unique(ArrayList<T> list, Boolean reversible) {
		ArrayList<T> copied = new ArrayList<T>(Set.copyOf(list));		
		
//		HashMap<T, Integer> mapping = new HashMap<>();
//		if (reversible) {
//			for (T element: list) {
//				if (!mapping.containsKey(element)) {
//					mapping.put(element, list.indexOf(element));
//				}
//			}
//		}
		
		copied.sort(null);
//		ArrayList<Object> container = new ArrayList<Object>();
//		container.add(copied);
		ArrayList<Object> container = Stream.of(copied).collect(Collectors.toCollection(ArrayList::new));
		
		if (reversible) {
			int[] ordering = list.stream().mapToInt(copied::indexOf).toArray();
			container.add(ordering);
		}
		
		return container.toArray();
	}
	
	private static int[] bincount(List<Integer> bins) {
		ArrayList<Integer> bincount = IntStream.range(0, bins.size()).map(element -> Collections.frequency(bins, element)).collect(ArrayList::new, (accumulator, element) -> accumulator.add(element), (collector, accumulator) -> collector.addAll(accumulator));
		return bincount.stream().filter(i -> i != 0).mapToInt(Integer::intValue).toArray();
//		return bins.stream().mapToInt(element -> Collections.frequency(bins, element)).toArray();
	}
	
	@SuppressWarnings("unchecked")
	private static <T> ArrayList<T> permutation(Either<Integer, ArrayList<T>> range, Random state) {
		ArrayList<T> container = new ArrayList<>();
		if (range.get() instanceof Integer) {
			container = (ArrayList<T>) IntStream.range(0, range.get()).boxed().collect(Collectors.toCollection(ArrayList::new));
		} else
			container = new ArrayList<T>(range.get());
		Collections.shuffle(container, state);
		return container;
	}

	private static List<Integer> indices(ArrayList<?> list, int[] target_variable) {
//		TODO: write a more robust implementation
		return IntStream.range(0, list.size()).boxed().collect(Collectors.toList());
	}
	
	private static <T> ArrayList<int[]> masks(ArrayList<?> list, int[] target_variable) {
		ArrayList<int[]> masks = new ArrayList<>();
		for (int i : indices(list, target_variable)) {
			int[] mask = new int[list.size()];
			Arrays.fill(mask, 0);
			mask[i] = 1;
			masks.add(mask);
		}
		return masks;
	}
	
	public static ArrayList<int[][]> split_indices(ArrayList<?> list, int[] target_variable) {
		int[] indices = IntStream.range(0, list.size()).toArray();
		ArrayList<int[][]> split_indices = new ArrayList<>();
		for (int[] mask : masks(list, target_variable)) {
			int[] training = IntStream.range(0, indices.length).filter(i -> mask[i] != 1).toArray();
			int[] testing = {Arrays.stream(mask).filter(i -> i == 1).findFirst().getAsInt()};
			split_indices.add(new int[][]{training, testing});
		}
		return split_indices;
	}

	public static int[] take_from_indices(ArrayList<Integer> list, ArrayList<Integer> indices, String mode) {
		ArrayList<Integer> cloned_indices = new ArrayList<>(indices);
		switch(mode) {
			case "raise": {
				if (indices.parallelStream().anyMatch(i -> i >= list.size())) {
					throw new ArrayIndexOutOfBoundsException("indices::{%s} contains at least one index that exceeds [list.size()]".formatted(indices));
				}
			}
			case "wrap": {
//				NOTE: wrap every out-of-bounds index around [list.length]
				for (int i = 0; i < cloned_indices.size(); ++i) {
					cloned_indices.set(i, (cloned_indices.get(i) % list.size()));
				}				
				break;
			}
			case "clip": {
//				NOTE: replace every out-of-bounds index with [list.length - 1]
				for (int i = 0; i < cloned_indices.size(); ++i) {
					cloned_indices.set(i, Math.max(0, Math.min(cloned_indices.get(i), list.size() - 1)));
				}
				break;
			}
			default: throw new InvalidParameterException("mode::{%s} must be valid mode from {'raise', 'wrap', 'clip'}".formatted(mode));
		}
		return cloned_indices.stream().mapToInt(i -> list.get(i)).toArray();
	}
		
	public static Boolean exists(List<?> list) {
		return (list != null);
	}
	
	public static Boolean exists_and_is_empty(List<?> list) {
		if (exists(list)) {
			if (list.isEmpty()) { return true; }
		}
		return false;
	}
	
	public static Boolean exists_and_is_non_empty(List<?> list) {
		if (exists(list)) {
			if (!list.isEmpty()) { return true; }
		}
		return false;
	}
	
	public static <T> ArrayList<T> subsection_of(List<T> list, int from, int to) {
		return list.subList(from, to).parallelStream().collect(Collectors.toCollection(ArrayList::new));
	}
	
	@SuppressWarnings("unchecked")
	public static <T> ArrayList<T>[] allocate(ArrayList<T> list, double percentage_for_training, Either<Random, Integer> state, Boolean shuffle, ArrayList<?> stratified) {
//		TODO: Range(0, "<=").and(1, ">=").contains(test_size);
		if (percentage_for_training <= 0 || percentage_for_training >= 1) {
			throw new 
				IllegalArgumentException("percentage_for_training::{%f} must be in Range(0, 1)".formatted(percentage_for_training));
		}
		int training_portion = (int)Math.floor(percentage_for_training * list.size());
		int testing_portion = list.size() - training_portion;

		
		if (training_portion == 0) {
			throw new 
				InvalidParameterException("With {%d}::samples, test_size::{%f} and train_size::{%f}, the resulting train set will be empty. Adjust any of the aforementioned parameters".formatted(list.size(), 1 - percentage_for_training, percentage_for_training));
		}
		
		if (shuffle == false) {
			ArrayList<T>[] splices = new ArrayList[2];
			if (exists_and_is_empty(stratified)) {
				throw new InvalidParameterException("Cannot stratify list if [shuffle == false]");
			}

			ArrayList<T> training_slice = subsection_of(list, 0, training_portion);
			ArrayList<T> testing_slice = subsection_of(list, training_portion, training_portion + testing_portion);
			splices = new ArrayList[] {training_slice, testing_slice};
			return splices;
		} else {
			final int splits = 10;
			if (exists_and_is_non_empty(stratified)) {
				if (stratified.size() != list.size())
					throw new InvalidParameterException("[list] and [stratify] labels are of inconsistent lenghts");
					
				if (stratified.get(0) instanceof List<?>) {
//					ArrayList<String> labels = stratify.stream().map(row -> row.stream().map(Object::toString).collect(Collectors.joining(" "))).collect(Collectors.toCollection(ArrayList::new));
				}
				
				Object[] uniqued = unique(stratified, true);
				ArrayList<?> classes = (ArrayList<?>) uniqued[0];
				int[] stratified_indices = (int[]) uniqued[1];
				
				int[] class_counts = bincount(Arrays.stream(stratified_indices).boxed().collect(Collectors.toList()));
				if (Arrays.stream(class_counts).min().getAsInt() < 2) {					
					throw new InvalidParameterException("The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.");
				}
				
				if (training_portion < classes.size()) {					
					throw new InvalidParameterException("The train_size::{%d} should be >= the number of classes::{%d}".formatted(training_portion, classes.size()));
				}
				
				if (testing_portion < classes.size()) {					
					throw new InvalidParameterException("The test_size::{%d} should be >= the number of classes::{%d}".formatted(testing_portion, classes.size()));
				}
								
				ArrayList<ArrayList<Integer>> class_indices = split_into(IntStream.range(0, stratified_indices.length).boxed().collect(Collectors.toCollection(ArrayList::new)), class_counts.length);
				Random rng = check_random_state(state.get());
				
				ArrayList<Integer> train = new ArrayList<>(), test = new ArrayList<>();
				ArrayList<Integer>[] indices = new ArrayList[2];
				for (int __ = 0; __ < 10; ++__) {
//						https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/model_selection/_split.py#L1977
					int[] _n = MathUtils.MultivariateHypergeometricDistribution.approximate_mode(class_counts, training_portion, rng);
					int[] remaining = IntStream.range(0, class_counts.length).map(i -> class_counts[i] - _n[i]).toArray();
					int[] _t = MathUtils.MultivariateHypergeometricDistribution.approximate_mode(remaining, testing_portion, rng);
					for (int i = 0; i < classes.size(); ++i) {
						ArrayList<Integer> permutation = permutation(Either.create_with(class_counts[i]), rng);
//							https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/model_selection/_split.py#L2108
						int[] permuted_class_indices = take_from_indices((ArrayList<Integer>)class_indices.get(i), permutation, "clip");

						train.addAll(
//									NOTE: training_slice
								Arrays.stream(Arrays.copyOfRange(permuted_class_indices, 0, _n[i]))
									  .boxed()
									  .collect(Collectors.toCollection(ArrayList::new))
						  );

						test.addAll(
//									NOTE: testing_slice
								Arrays.stream(Arrays.copyOfRange(permuted_class_indices, _n[i], _n[i] + _t[i]))
									  .boxed()
									  .collect(Collectors.toCollection(ArrayList::new))
						  );
					}
					train = permutation(Either.create_with(train), rng);
					test = permutation(Either.create_with(test), rng);

					indices[0] = train;
					indices[1] = test;
					
					train = new ArrayList<>();
					test = new ArrayList<>();
				}
				return Arrays.stream(indices)
							 .map(i -> {
								 return i.parallelStream()
										 .map(e -> list.get(e))
										 .collect(Collectors.toCollection(ArrayList::new));
							 }).toArray(ArrayList[]::new);
			}		
			
			ArrayList<T>[] splices = new ArrayList[2];
			Random rng = check_random_state(state.get());
			for (int __ = 0; __ < splits; ++__) {
				ArrayList<T> permutation = permutation(Either.create_with(list), rng);
				ArrayList<T> fortesting = subsection_of(permutation, 0, testing_portion);
				ArrayList<T> fortraining = subsection_of(permutation, testing_portion, (testing_portion + training_portion));
				if (__ + 1 == splits)
					splices = new ArrayList[]{fortraining, fortesting};
			}
			return splices;
//			https://github.com/scikit-learn/scikit-learn/blob/364c77e04/sklearn/model_selection/_split.py#L2586
		}
	}

	
	public static class Position<F extends Number, S extends Number> implements Comparable<Position> {
		public F x;
		public S y;
		
		Position(F first, S second) { this.x = first; this.y = second; }
		static <F extends Number, S extends Number> Position make_position(F first, S second) {
			return new Position<F, S>(first, second);
		}
		
		@Override
		public int compareTo(Position other) {
			if (this.x == other.x)
				return this.y.intValue() - other.y.intValue();
			else {
				return this.x.intValue() - other.x.intValue();
			}
		}
		
		@Override
		public String toString() {
			return String.format("(%s, %s)", x, y);
		}
		
		public double minimum_distance_to(Position other) {
			return Math.sqrt(
					Math.pow(this.x.doubleValue() - other.x.doubleValue(), 2)
					+
					Math.pow(this.y.doubleValue() - other.y.doubleValue(), 2)
			);
		}
		
		public <T extends Number> Position<F, S> plus(T scalar) {
//			TODO: add scalar in a better way
			return Position.make_position(x.doubleValue() + scalar.doubleValue(), y.doubleValue() + scalar.doubleValue());
		}
	}
	
//    2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
//    Returns a positive value, if OAB makes a counter-clockwise turn,
//    negative for clockwise turn, and zero if the points are collinear.
	private static long magnitude_of_cross_product_in_2D_space(Position<Integer, Integer> P, Position<Integer, Integer> Q, Position<Integer, Integer> R) {
		return (Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y);
	}
	
	public static long dot_product(Position<Integer, Integer> P, Position<Integer, Integer> Q, Position<Integer, Integer> R) {
		return (Q.x - P.x) * (R.x - Q.x) + (Q.y - P.y) * (R.y - Q.y);
	}
	
//  https://stackoverflow.com/questions/2736290/how-to-find-two-most-distant-points
//	https://stackoverflow.com/questions/243945/calculating-a-2d-vectors-cross-product
//	https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain#Java
	public static Position<Integer, Integer>[] convex_hull(Position<Integer, Integer>[] positions) {
//		Computes the convex hull of a set of 2D points.
//
//	    Input: an ArrayList of Positions representing the points.
//	    Output: a list of vertices of the convex hull in counter-clockwise order,
//	      starting from the vertex with the lexicographically smallest coordinates.
//	    Implements Andrew's monotone chain algorithm. O(n log n) complexity.

		Position<Integer, Integer>[] convex_hull_set = (Position<Integer, Integer>[]) new Position[2 * positions.length];
		Arrays.sort(positions);
int k = 0;
		
//		Build lower hull
		int i = 0;
		for ( ; i < positions.length; ++i) {
			while ((k >= 2) && (magnitude_of_cross_product_in_2D_space(convex_hull_set[k - 2], convex_hull_set[k - 1], positions[i]) <= 0))
				k--;
			convex_hull_set[k++] = positions[i];
		}
		
//		Build upper hull
		int t = 0;
		for (i = positions.length - 2, t = k + 1; i >= 0; i--) {
			while ((k >= t) && (magnitude_of_cross_product_in_2D_space(convex_hull_set[k - 2], convex_hull_set[k - 1], positions[i]) <= 0))
				k--;
			convex_hull_set[k++] = positions[i];
		}
		
		if (k > 1)
			convex_hull_set = Arrays.copyOfRange(convex_hull_set, 0, k - 1); // remove non-hull vertices after k; remove k + 1 which is a duplicate
		
		return convex_hull_set;
	}
	
	
//	https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
	public static double minimum_distance_to_segment(Position<Integer, Integer> first_line_segment_sample, Position<Integer, Integer> second_line_segment_sample, Position<Integer, Integer> point) {
		// Return minimum distance between the line segment formed from {first_line_segment_sample} and {second_line_segment_sample} and a {point}
		
		double segment_length = first_line_segment_sample.minimum_distance_to(second_line_segment_sample);
		if (segment_length == 0) {
			return first_line_segment_sample.minimum_distance_to(point);
		}
		
		double projection_distance = dot_product(first_line_segment_sample, second_line_segment_sample, point) / Math.pow(segment_length, 2);
		projection_distance = Math.max(0, Math.min(1, projection_distance));
		
		final Position<Integer, Integer> projection = Position.make_position(
				first_line_segment_sample.x + projection_distance * (second_line_segment_sample.x - first_line_segment_sample.x), 
				first_line_segment_sample.y + projection_distance * (second_line_segment_sample.y - first_line_segment_sample.y)
		);
		return Math.sqrt(point.minimum_distance_to(projection));
	}
	
//	https://web.archive.org/web/20180528104521/http://www.tcs.fudan.edu.cn/~rudolf/Courses/Algorithms/Alg_ss_07w/Webprojects/Qinbo_diameter/2d_alg.htm
//	https://web.archive.org/web/20170620040641/http://www2.seas.gwu.edu/~simhaweb/alg/lectures/module1/module1.html
	public static <F extends Number, S extends Number> double maximum_distance_between_positions(Position<Integer, Integer>[] convex_hull_set) {
		if (convex_hull_set.length == 2) {
			return convex_hull_set[0].minimum_distance_to(convex_hull_set[1]);
		}
		
		ArrayList<Position<Integer, Integer>[]> antipodal_pairs = new ArrayList<>();
int k = 2; // farthest vertex index from line made from first and last vertex: is at bottom
		int last_index = convex_hull_set.length - 1;
		while (minimum_distance_to_segment(convex_hull_set[last_index], convex_hull_set[0], convex_hull_set[k + 1]) > minimum_distance_to_segment(convex_hull_set[last_index], convex_hull_set[0], convex_hull_set[k]))
			k++;

int i = 0, j = k;
		while (i <= k && j <= last_index) {
			antipodal_pairs.add(new Position[] {convex_hull_set[i], convex_hull_set[j]});
			while (j < last_index && minimum_distance_to_segment(convex_hull_set[i], convex_hull_set[i + 1], convex_hull_set[j + 1]) > minimum_distance_to_segment(convex_hull_set[i], convex_hull_set[i + 1], convex_hull_set[j])) {
				antipodal_pairs.add(new Position[] { convex_hull_set[i], convex_hull_set[j] });
				j++;
			}
			i++;
		}
		
		double largest_distance = maximum_distance_between_positions(antipodal_pairs.get(0));
		for (Position<Integer, Integer>[] antipodal_pair : antipodal_pairs) {
			if (maximum_distance_between_positions(antipodal_pair) > largest_distance) {
				largest_distance = maximum_distance_between_positions(antipodal_pair);
			}
		}
		return largest_distance;
	}
}
