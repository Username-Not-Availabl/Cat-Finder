package com.csc.honors_module;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

public class ProgressBarRunnable implements Runnable {

	private boolean stop = false;
	
	private int total;
	private Incrementer incrementer;
	
	public void stop() {
		this.stop = true;
		this.progressBar(total, total);
	}

	private synchronized boolean _continue() {
		return this.stop == false;	
	}

	@Override
	public void run() {
		while (_continue()) {
			synchronized (this) {
				if (incrementer.getValue() == total)
					this.stop();
				else {					
					this.progressBar(incrementer.getValue(), total);
				}
			}
//			this.progressBar(, total);
//			https://bitek.dev/blog/java_threading_shared_data_tutorial/
//			https://www.google.com/search?q=share+data+between+threads+java&oq=share+values+among+threads+java&aqs=chrome.1.69i57j0i22i30l2j0i390i650l3.15247j0j7&sourceid=chrome&ie=UTF-8
			
//			https://stackoverflow.com/questions/9148899/returning-value-from-thread
			try {
				Thread.sleep(500);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	ProgressBarRunnable(int total, Incrementer incrementer) {
		this.total = total;
		this.incrementer = incrementer;
	}
	
	private static int cached_response = -1;
	public static int extend_width_to_fill_console() throws IOException {
		if (cached_response != -1) {
			return cached_response;
		}
		
		String os = System.getProperty("os.name").toLowerCase();
		String[] command = new String[2];
		if (os.contains("win")) {
			command = new String[] {"pwsh", "-Command", "$Host.UI.RawUI.WindowSize.Width"};
		}
		
		if (Arrays.stream(new String[] {"mac", "linux"}).anyMatch(element -> os.contains(element))) {
//			command = new String[] {"bash", "-c", "tput cols 2> /dev/tty" };
			command = new String[] {"bash", "tput cols 2> /dev/tty" };
		}
		
		if (command[0] == Debug.NULL()) {
			throw new IOException("[ERROR]: unfamiliar operating system");
		}
		Process process = Runtime.getRuntime().exec(command);
		BufferedReader std_input = new BufferedReader(new InputStreamReader(process.getInputStream()));

		String line = null;
		while((line = std_input.readLine()) != null) {break;} // Note: should be first line

		process.destroy();
		return Integer.valueOf(line);
	}
	
//  https://stackoverflow.com/questions/852665/command-line-progress-bar-in-java
	public synchronized void progressBar(int quotient, int total) {
//		System.out.printf("quotient:: %d | total:: %d  ", quotient, total);
		if (quotient > total)
			throw new IllegalArgumentException("[ERROR]: quotient cannot be greater than total");
		
		int maximum = 50;
		try {
			maximum = extend_width_to_fill_console();
		} catch (IOException e) {};
		int percentage = (int) (((float)(quotient * maximum)) / (total));
		
		char unloaded = '-';
		String loaded = "#";
		String bar = new String(new char[maximum]).replace('\0', unloaded) + "]";
		StringBuilder part_finished = new StringBuilder();
		part_finished.append("[");
		for (int i = 0; i < percentage; ++i) {
			part_finished.append(loaded);
		}
		
		String part_remaining = bar.substring(percentage, bar.length());
		System.out.printf("\r%s%s |%.2f%%", part_finished, part_remaining, 100 * ((float)percentage / maximum));
		if (quotient == total)
			System.out.println();
	}

}
