package com.csc.honors_module;

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
	
//  https://stackoverflow.com/questions/852665/command-line-progress-bar-in-java
	public synchronized void progressBar(int quotient, int total) {
//		System.out.printf("quotient:: %d | total:: %d  ", quotient, total);
		if (quotient > total)
			throw new IllegalArgumentException("[ERROR]: quotient cannot be greater than total");
		
		int maximum = 50;
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
