package com.csc.honors_module;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Stroke;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.Timer;

import com.csc.honors_module.MathUtils.Random;
import com.csc.honors_module.ModelUtils.Position;
import com.csc.honors_module.Range.infinitum;

@SuppressWarnings("unchecked")
public class GUI {

	private static final int MILISECOND = 1;
	private static final int SECOND = 1000 * MILISECOND;
	
	private static int BORDER_GAP = 25; // (used to be: 30)
	private static int LABEL_PADDING = 25;
	
	public static int MAXIMUM = 100;
	public static int MINIMUM = 0;
	private static Position<Integer, Integer> PREFERRED_SIZE = Position.make_position(800, 650);

	
	public static class DynamicGraph extends JPanel {
		private static final long serialVersionUID = -8140289861636092075L;

		private final List<Double> dataPoints = new ArrayList<>();
		private Boolean _continue = true;

		private static Color GRID_COLOR = new Color(200, 200, 200, 200);
		private static Color STROKE_COLOR = new Color(44, 102, 230, 180); //Color.ORANGE;
		private static Color POINT_COLOR = new Color(150, 50, 50, 180);

		private static Stroke STROKE = new BasicStroke(3f);
//		private static int POINT_WIDTH = 12;
		private static int POINT_WIDTH = 2;
		
		private static int NUMBER_OF_HATCHES = 10; // (percent markers)
		private static int YDIVISIONS = 10;
		
		
		public void addDataPoint(final int x, final double y) {
			dataPoints.add(Math.max(MINIMUM + 5, y));
			this.repaint();
		}
		
		@Override
		protected void paintComponent(final Graphics __graphics) {
			super.paintComponent(__graphics);
			final Graphics2D graphics = (Graphics2D) __graphics;
			graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			
			final Position<Double, Double> scale = Position.make_position(
					((double) this.getWidth() - (2 * BORDER_GAP) - LABEL_PADDING) / (this.dataPoints.size() - 1),
					((double) this.getHeight() - 2 * BORDER_GAP) / (MAXIMUM - 1)
			);
			
			final List<Position<Integer, Integer>> positions = new ArrayList<>();
			for (int i = 0; i < this.dataPoints.size(); i++) {
				if (dataPoints.get(i) <= 5) {
					this._continue = false;
				}
				positions.add(Position.make_position(
					(int) (i * scale.x + BORDER_GAP + LABEL_PADDING),
					(int) ((MAXIMUM - this.dataPoints.get(i)) * scale.y + BORDER_GAP)
				));
			}
			
			graphics.setColor(Color.WHITE);
			graphics.fillRect(BORDER_GAP + LABEL_PADDING, BORDER_GAP, this.getWidth() - (2 * BORDER_GAP) - LABEL_PADDING, this.getHeight() - (2 * BORDER_GAP) - LABEL_PADDING);
			graphics.setColor(Color.BLACK);
			
//			creates hatch marks and grid lines for y axis
			for (int i = 0; i < YDIVISIONS + 1; ++i) {
				final int y = this.getHeight() - ((i * (this.getHeight() - (2 * BORDER_GAP) - LABEL_PADDING)) / YDIVISIONS + BORDER_GAP + LABEL_PADDING);
				final Position<Integer, Integer> start = Position.make_position(BORDER_GAP + LABEL_PADDING, y);
				final Position<Integer, Integer> end = Position.make_position(POINT_WIDTH + BORDER_GAP + LABEL_PADDING, y);
				if (this.dataPoints.size() > 0) {
					graphics.setColor(GRID_COLOR);
	                graphics.drawLine(BORDER_GAP + LABEL_PADDING + 1 + POINT_WIDTH, start.y, this.getWidth() - BORDER_GAP, end.y);
	                graphics.setColor(Color.BLACK);

	                final String label = ((int) ((MINIMUM + (MAXIMUM - MINIMUM) * ((i * 1.0) / YDIVISIONS)) * 100)) / 100.0 + "";
	                final FontMetrics metrics = graphics.getFontMetrics();
	                
	                graphics.drawString(label, start.x - metrics.stringWidth(label) - 5, start.y + (metrics.getHeight() / 2) - 3);
				}
				graphics.drawLine(start.x, start.y, end.x, end.y);
			}
			
//			creates hatch marks and grid lines for x axis
			for (int i = 0; i < this.dataPoints.size(); ++i) {
				if (this.dataPoints.size() > 1) {
					final int x = i * (this.getWidth() - BORDER_GAP * 2 - LABEL_PADDING) / (this.dataPoints.size() - 1) + BORDER_GAP + LABEL_PADDING;
					final Position<Integer, Integer> start = Position.make_position(x, this.getHeight() - BORDER_GAP - LABEL_PADDING);
					final Position<Integer, Integer> end = Position.make_position(x, start.y - POINT_WIDTH);
					if ((i % ((int) ((this.dataPoints.size() / 20.0)) + 1)) == 0) {
						graphics.setColor(GRID_COLOR);
	                    graphics.drawLine(start.x, this.getHeight() - BORDER_GAP - LABEL_PADDING - 1 - POINT_WIDTH, end.x, BORDER_GAP);
						graphics.setColor(Color.BLACK);
						
						final String label = String.valueOf(i);
						final FontMetrics metrics = graphics.getFontMetrics();
						
	                    graphics.drawString(label, start.x - metrics.stringWidth(label) / 2, start.y + metrics.getHeight() + 3);
					}
					graphics.drawLine(start.x, start.y, end.x, end.y);
				}
			}
			
//			creates axes lines
			graphics.drawLine(BORDER_GAP + LABEL_PADDING, this.getHeight() - BORDER_GAP - LABEL_PADDING, BORDER_GAP + LABEL_PADDING, BORDER_GAP);
	        graphics.drawLine(BORDER_GAP + LABEL_PADDING, this.getHeight() - BORDER_GAP - LABEL_PADDING, this.getWidth() - BORDER_GAP, this.getHeight() - BORDER_GAP - LABEL_PADDING);
	        
	        final Stroke cached = graphics.getStroke();
	        graphics.setColor(STROKE_COLOR);
	        graphics.setStroke(STROKE);
	        for (int i = 0; i < positions.size() - 1; ++i) {
	        	final Position<Integer, Integer> start = positions.get(i);
	        	final Position<Integer, Integer> end = positions.get(i + 1);
	        	graphics.drawLine(start.x, start.y, end.x, end.y);
	        }
	        
	        graphics.setStroke(cached);
	        graphics.setColor(POINT_COLOR);
	        for (int i = 0; i < positions.size(); ++i) {
	        	graphics.fillOval(
	        			positions.get(i).x - POINT_WIDTH / 2, 
	        			positions.get(i).y - POINT_WIDTH / 2, 
	        			POINT_WIDTH, POINT_WIDTH);
	        }
		}
				
		@Override
		public Dimension getPreferredSize() {
			return new Dimension(PREFERRED_SIZE.x, PREFERRED_SIZE.y);
		}
		
		protected Boolean isFinished() {
			return !this._continue;
		}
		
		public static void instantiate_with(final String title, final Supplier<Double> final_synchronized_supplier) {
			final DynamicGraph graph = new DynamicGraph();

			final JFrame frame = new JFrame(title);
			frame.addKeyListener((KeyListener) new KeyAdapter() {
				@Override
				public void keyPressed(final KeyEvent event) {
					if (event.getKeyCode() == KeyEvent.VK_Q) {
						System.exit(0);
					}
				}
			});
			frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
			frame.getContentPane().add(graph);
			frame.pack();
			frame.setLocationRelativeTo(null);
			frame.setVisible(true);
						
			final Timer timer = new Timer(1 * MILISECOND, new ActionListener() {
				private int x = 0; //TODO: sample Model for error percentage
				
				@Override
				public void actionPerformed(final ActionEvent e) {
					if (!graph.isFinished()) {
						graph.addDataPoint(x++, final_synchronized_supplier.get());
					}
				}
			});
			timer.start();
		}
		
		private static AtomicReference<Double> mutable_reference = new AtomicReference<>();
		public synchronized static void setReference(Double value) {
			mutable_reference.set(value);
		}
		
//		NOTE: meant to serve as Overridable method
		public synchronized static Double __increment() {
			System.out.println(mutable_reference.get());
			return mutable_reference.get();
		}
		
		public static void execute_as(final String title) {
			SwingUtilities.invokeLater(new Runnable() {
				@Override
				public void run() {
					GUI.DynamicGraph.instantiate_with(title, () -> {
//						return mutable_reference.get();
						return __increment();
					});
				}
			});
		}
	}
	
	private static double e = 100;
	public static void update() {
		GUI.e = GUI.e - Random.between(Range.of(infinitum.FALSE, 0.0, 5.0));
	}
	
	public static double __decrease() {
//		GUI.e = GUI.e - Random.between(Range.of(infinitum.FALSE, 0.0, 5.0));
		update();
		return GUI.e;
	}
	
	
	public static void main(final String ...args) {
		
//		double e = 100.0;
		SwingUtilities.invokeLater(new Runnable() {
			@Override
			public void run() {
				GUI.DynamicGraph.instantiate_with("Learning Rate", () -> {
					return __decrease();
				});
			}
		});
		
//		double e = 1000000000;
//		double e = 0;
//		AtomicReference<Double> ref = new AtomicReference<Double>(e);
////		ScheduledExecutorService executor = Executors.newScheduledThreadPool(1);
//		GUI.DynamicGraph.setReference(e);
//		GUI.DynamicGraph.execute_as("Test");
//		for (; e < 100; e += 5) {
////			System.out.println(e);
//			GUI.DynamicGraph.setReference(e);
//			
////			e = e / 10;
//		}
		
//		executor.scheduleAtFixedRate(null, 0, 1, TimeUnit.SECONDS);
	}
}

