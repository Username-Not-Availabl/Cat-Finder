package com.csc.honors_module;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import org.opencv.core.Mat;

import com.csc.honors_module.MathUtils.Activation;

public class Debug {	
	public static <T> T unimplemented() {
		return null;
	}
	
	public static String printc(String format, Object ...args) {
		String printable = "[DEBUG]::{%s}\n".formatted(String.format(format, args));
		for (Object arg: args) {
			if ((arg instanceof String) && arg.equals("mute")) {				
				return printable;
			}			
		}
		System.out.printf(printable);
		return null;
	}
	
	public static void prints(String format) {
		Debug.printc(format, new Object[0]);
	}
	
	public static void printo(Object format) {
//		if (format instanceof Collection) {
//			Debug.print(new Object[]{format});
//			return;
//		}
		Debug.prints(format.toString());
	}
	
	
	public static <T> List<String> collect(List<int[][]> list) {
		if (list.get(0).getClass().isArray()) {
			return list.parallelStream().map(i -> Arrays.toString(Arrays.stream(i).map(e -> Arrays.toString(e)).toArray())).collect(Collectors.toList());
		}
		return null;
	}
	
//	private static HashMap<String, Integer> records = new HashMap<>();
	public static void printSurrounded(String header, Object arg, Consumer<Mat> ...printer) {
		final int LENGTH = 35;
		String padding = "=".repeat(LENGTH).substring(header.length());
		String formatted = padding + header + padding;
		System.out.println(formatted);
		if (printer.length == 1)
			printer[0].accept((Mat)arg);
		else
			Debug.print(arg);
		System.out.println(padding + "END" + padding);
	}
	
	public static void print3DM(Mat matrix) {
		for (int i = 0; i < matrix.size(0); i++) {
			for (int j = 0; j < matrix.size(1); j++) {
				for (int k = 0; k < matrix.size(2); k++) {
					double[] at = matrix.get(new int[] { i, j, k });
					System.out.print(Arrays.toString(at) + " ");
				}
				System.out.println();
			}
			if (i != matrix.size(0) - 1) {
				System.out.println();
				System.out.println();
			}
		}

	}
	
	public static void print(Object ...args) {
		StringBuilder builder = new StringBuilder(args.length > 1 ? "\n" : " ");
		for (Object arg: args) {
			if (arg == null) {
				builder.append("!(null)");
				continue;
			}
			
			if (arg instanceof Collection) {
//				String a = Debug.printc(arg.toString(), "mute");
				if (arg instanceof List<?>) {
					List<int[][]> asList = (List<int[][]>) arg;
//					System.out.println(Arrays.toString(ArrayList.class.getFields()));
//					builder.append("\n\t" + Debug.printc(asList.toString(), "mute"));
					try {
						List<String> asListOfString = collect(asList);
						builder.append("\n");
						asListOfString.parallelStream().forEachOrdered(i -> builder.append("\t" + i + "\n"));						
					} catch (Exception exception) {
						builder.append(arg.toString());
					};
				}
				continue;
				
			}

			if (arg.getClass().isArray()) {
				Object[] asObjectArray = new Object[0];
				try {
					asObjectArray = (Object[])arg;
					builder.append("%s|%s".formatted(args.length > 1 ? "\t" : " ", Arrays.toString(asObjectArray)));
				} catch(ClassCastException exception) {
					if (arg instanceof int[]) {
						builder.append("%s|%s".formatted(args.length > 1 ? "\t" : " ", Arrays.toString((int[])arg)));
					}
				}
				continue;
			}

			if (arg instanceof Mat) {
				builder.append("%s|\n%s".formatted(((Mat)arg).toString(), ((Mat)arg).dump() ));
				continue;
			}
			builder.append("%s|%s".formatted(args.length > 1 ? "\t" : " ", arg.toString()));
		}
		builder.append(args.length > 1 ? "\n" : " ");
		Debug.prints(builder.toString());
		
	}

	public static <T> T NULL() {
		return null;
	}
}
