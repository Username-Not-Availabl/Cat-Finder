package com.csc.honors_module;

import java.security.InvalidParameterException;
import java.util.Arrays;
import java.util.Objects;

import com.csc.honors_module.ModelUtils.Either;

public class Range<T extends Number> {
	public static enum infinitum {
		FALSE("FALSE"), POSITIVE("POSITIVE"), NEGATIVE("NEGATIVE");
		private String string_representation;
		
		infinitum(String state) {string_representation = state;}
		
		public String toString() {return string_representation;}
		public Boolean is_infinitum() {return true;}
	};
	
	Either<T, infinitum> from;
	Either<T, infinitum> to;
	
	private <E extends Number> Range() {}
	@SafeVarargs
	public static <E extends Number> Range<E> of(infinitum state, E ...numbers) {
		Objects.requireNonNull(numbers);
		if (numbers.length > 2) {
			throw new InvalidParameterException("Range must be between two Endpoints");
		}

		Range<E> instance =  new Range<E>();
		switch (state) {
			case FALSE: {				
				if (numbers.length == 2) {
					instance.from = Either.create_with(numbers[0]);
					instance.to = Either.create_with(numbers[1]);
					return instance;
				}
				throw new InvalidParameterException("Insufficient number of Endpoints for state::{%s}".formatted(state));
			}
			case POSITIVE: {
				instance.from = Either.create_with(numbers[0]);
				instance.to = Either.create_with(state);
			}
			case NEGATIVE: {
				instance.from = Either.create_with(state);
				instance.to = Either.create_with(numbers[0]);
			}
		};
		return instance;
	}
	
	public <T extends Number> Boolean contains(T number, Boolean inclusive) {
		Boolean[] comparison = new Boolean[2];		
		if (!(this.from.get() instanceof infinitum)) {
			comparison[0] = ((Number)this.from.get()).doubleValue() < number.doubleValue();
			comparison[0] |= (((Number)this.from.get()).doubleValue() == number.doubleValue() && inclusive);
		}
		
		if (!(this.to.get() instanceof infinitum)) {
			comparison[1] = ((Number)this.to.get()).doubleValue() > number.doubleValue();
			comparison[1] |= (((Number)this.to.get()).doubleValue() == number.doubleValue() && inclusive);
		}
		return !Arrays.stream(comparison)
					  .filter(element -> element != null)
					  .anyMatch(element -> element == false);
	}
	
	public static void main(String[] args) {
//		Range<Integer> range = Range.of(Range.infinitum.NEGATIVE, 1);
		
//		System.out.println(range.contains(1, true));
		System.out.print("=====================Range========================");
	}
}
