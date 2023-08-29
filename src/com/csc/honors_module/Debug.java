package com.csc.honors_module;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Scanner;
import java.util.Stack;
import java.util.function.Consumer;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.opencv.core.Mat;

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
	
	final private static String UserSpace = System.getProperty("user.dir") + "\\src\\com\\csc\\honors_module\\";
	public static void printNamed(Object ...printables) {
		StackTraceElement[] elements = Thread.currentThread().getStackTrace();
		StackTraceElement callerContext = elements[2];
		
		try {
			Scanner scanner = new Scanner(new File(UserSpace + callerContext.getFileName()));
			String data;
			int i = 1;

			String self = elements[1].getMethodName();
			int offset = (self + "(").length();

			Stack<Character> stack = new Stack<Character>();
			stack.push('(');
			String contents = "";
			ArrayList<String> arguments = new ArrayList<>();
			while ((data = scanner.nextLine()) != null) {
				if (i >= callerContext.getLineNumber()) {
					int start = data.indexOf(self + "(") + offset;
					String line = data.substring(start);
					contents += line + '\n';
											
					for (char current: line.toCharArray()) {
						if (current == '(') { stack.push(current); }
						else if (current == ')') {
							if (stack.isEmpty() || stack.peek() != '(') {
								break;
							}
							stack.pop();
						}
						
						if (current == ',' && stack.size() == 1) {														
							String[] split = line.split(",", 2);
							arguments.add(contents.replaceFirst("(?s)(.*)" + Pattern.quote(split[1]), "$1" + "").trim());
							contents = split[1] + "\n";
							
							line = split[1];
						}
					}
					
					if (stack.isEmpty()) {
						arguments.add(contents.trim());
						break;
					} else if (stack.peek() == '(') {
						offset = 1;
					}
				}
				++i;
			}

			scanner.close();			
			
			ArrayList<String> __ = new ArrayList<>();
			__ = IntStream.range(0, arguments.size()).mapToObj((index) -> {
				String argument = arguments.get(index);
				if (index != arguments.size() - 1) {
					argument = argument.substring(0, argument.length() - 1)
							.replace("\n", " ").replace("\t", "");
				} else {
					argument = argument.substring(0, argument.length() - 2)
							.replace("\n", "").replace("\t", "");
				}
				
				return "<" + argument + ">";
			}).collect(Collectors.toCollection(ArrayList<String>::new));
			
			Debug.print(printables.length, printables);
			for (i = 0; i < printables.length; ++i) {
				Debug.print(__.get(i) + " => " + Debug.print("![printable]", printables[i]));
			}
		} catch (FileNotFoundException | NoSuchElementException e) {Debug.print("--false");}
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
	
	public static ArrayList<String> print(Object ...args) {
		StringBuilder builder = new StringBuilder(args.length > 1 ? "\n" : " ");
		ArrayList<String> results = new ArrayList<>();
		Boolean print = true;
		for (Object arg: args) {
			if (arg == null) {
				builder.append("!(null)");
				results.add("!(null)");
				continue;
			}
			
			if (arg == "![printable]") {
				print = false;
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
						asListOfString.parallelStream().forEachOrdered(i -> {
							builder.append("\t" + i + "\n");
							results.add("\t" + i + "\n");
						});						
					} catch (Exception exception) {
						builder.append(arg.toString());
						results.add(arg.toString());
					};
				}
				continue;
				
			}

			if (arg.getClass().isArray()) {
				Object[] asObjectArray = new Object[0];
				try {
					asObjectArray = (Object[])arg;
					builder.append("%s|%s".formatted(args.length > 1 ? "\t" : " ", Arrays.toString(asObjectArray)));
					results.add(Arrays.toString(asObjectArray));
				} catch(ClassCastException exception) {
					if (arg instanceof int[]) {
						builder.append("%s|%s".formatted(args.length > 1 ? "\t" : " ", Arrays.toString((int[])arg)));
						results.add(Arrays.toString((int[])arg));
					}
					
//					if (arg instanceof Mat[]) {
//						builder.append("%s|%s".formatted(args.length > 1 ? "\t" : " ", Arrays.toString((Mat[])arg)));
//						results.add(Arrays.toString((Mat[])arg));
//					}
				}
				continue;
			}

			if (arg instanceof Mat) {
				builder.append("%s|\n%s".formatted(((Mat)arg).toString(), ((Mat)arg).dump() ));
				results.add("%s|\n%s".formatted(((Mat)arg).toString(), ((Mat)arg).dump()));
				continue;
			}
			builder.append("%s|%s".formatted(args.length > 1 ? "\t" : " ", arg.toString()));
			results.add("%s|%s".formatted(args.length > 1 ? "\t" : " ", arg.toString()));
		}
		builder.append(args.length > 1 ? "\n" : " ");
		if (print)
			Debug.prints(builder.toString());
		return results;
	}

	public static <T> T NULL() {
		return null;
	}
}
