package com.csc.honors_module;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

public class Debug {
//	private static class Array {
//		int[] intarray;
//		
//		public static Array of(int[] intarray) {
//			return new Array(intarray);
//		}
//		
////		public static String contents() {
//////			return Arrays.toString
////		}
//		
//		private Array(int[] intarray) {}
//	}
	
	public static String printc(String format, Object ...args) {
		String printable = "[DEBUG]::{%s}\n".formatted(String.format(format, args));
		for (Object arg: args) {
			if ((arg instanceof String) && arg.equals("mute")) {				
				return printable;
			}
			
//			if ()
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
	
	public static void print(Object ...args) {
		StringBuilder builder = new StringBuilder(args.length > 1 ? "\n" : " ");
		for (Object arg: args) {
			if (arg instanceof Collection) {
//				String a = Debug.printc(arg.toString(), "mute");
				if (arg instanceof List<?>) {
					List<int[][]> asList = (List<int[][]>) arg;
//					System.out.println(Arrays.toString(ArrayList.class.getFields()));
//					builder.append("\n\t" + Debug.printc(asList.toString(), "mute"));

					List<String> asListOfString = collect(asList);
					builder.append("\n");
					asListOfString.parallelStream().forEachOrdered(i -> builder.append("\t" + i + "\n"));
				}
				continue;
				
			}

			if (arg.getClass().isArray()) {
				Object[] asObjectArray = new Object[0];
				try {
					asObjectArray = (Object[])arg;
				} catch(ClassCastException exception) {
					if (arg instanceof int[]) {
						builder.append("%s|%s".formatted(args.length > 1 ? "\t" : " ", Arrays.toString((int[])arg)));
					}
				}
				continue;
			}
			builder.append("%s|%s".formatted(args.length > 1 ? "\t" : " ", arg.toString()));
		}
		builder.append(args.length > 1 ? "\n" : " ");
		Debug.prints(builder.toString());
		
	}
}
