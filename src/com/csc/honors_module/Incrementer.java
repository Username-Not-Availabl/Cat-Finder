package com.csc.honors_module;

public class Incrementer {
	private int lock = 0;
	public int increment(int addend) {
		lock += addend;
		return lock;
	}
	
	public int getValue() {
		return lock;
	}
}
