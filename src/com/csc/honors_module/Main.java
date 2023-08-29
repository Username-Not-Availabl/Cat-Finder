package com.csc.honors_module;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import com.csc.honors_module.MathUtils.Activation;
import com.csc.honors_module.MathUtils.Loss;
import com.csc.honors_module.Model.Optimizer;
import com.csc.honors_module.ModelUtils.Either;


public class Main extends ClassLoader {
    public static void main(String[] arguments) throws IOException, InterruptedException, ClassNotFoundException {
	    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
         
//        String pathname = "training_media\\archive\\CAT_00\\00000001_000.jpg";
        String pathname = "archive\\CAT_00\\00000001_000.jpg";

//        test_of_allocate();
//        test_of_forward_propogate_DENSE();
//        test_of_XOR();
        
//        System.out.print(true);
//        Path boundaries = FileSystems.getDefault().getPath(pathname + ".cat", "");
//        BufferedImage image = ImageIO.read(new File(pathname));
//
//        Mat matrix = new Mat();
////        https://stackoverflow.com/questions/14958643/converting-bufferedimage-to-mat-in-opencv
//        byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
//        
//        Mat imageMatrix = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC3);
//        imageMatrix.put(0, 0, pixels);
//    	Imgproc.blur(imageMatrix, matrix, new Size(7, 7));
////     	System.out.println(String.format("3x3 Identity Matrix => %s", matrix.dump()));
//
//    	Image blurred = HighGui.toBufferedImage(imageMatrix);
//    	Main.display(image);
////    	Main.display(Main.toBufferedImage(blurred));

//        ---------------
//        Mat colour = Imgcodecs.imread(pathname);
//        
//        Mat gray = new Mat();
//        Mat draw = new Mat();
//        Mat wide = new Mat();
//        
//        Imgproc.cvtColor(colour, gray, Imgproc.COLOR_BGR2GRAY);
//        Imgproc.Canny(gray, wide, 50, 150, 3, false);
//        Imgproc.blur(colour, wide, new Size(7, 7));
//        wide.convertTo(draw, CvType.CV_8UC3);
//        
//        Main.display(Main.toBufferedImage(HighGui.toBufferedImage(draw)));
//        Main.display(Main.toBufferedImage(HighGui.toBufferedImage(colour)));
//        ----------------------------
        
//        Iterable<Path> filepath = FileSystems.newFileSystem(Paths.get("training_media"), getPlatformClassLoader()).getRootDirectories();
//        Iterator<Path> filepath = Paths.get("./training_media").iterator();

//        ArrayList<String> aux = new ArrayList<String>();
//        aux.add(pathname);
//        ArrayList<Path> paths = Main.collectImages(aux, Paths.get("training_media"));
//        ArrayList<BufferedImage> result = Main.normalizeImages(paths);
//        ---------------------

//        File serialized = new File("");
//        if (serialized.exists() && serialized.canRead()) {
//        	ObjectInputStream stream = new ObjectInputStream(new FileInputStream(serialized));
//        	Model model = (Model) stream.readObject();
//        }
        
        int DEBUG_COUNTER = 100;
        
        ArrayList<BufferedImage> normalized = new ArrayList<BufferedImage>();
        
//        TODO: make Main.collect images take Stream<Path> so all this work will be done inside
        try (Stream<Path> filepaths = Files.walk(Paths.get("training_media/archive"))) {
        	ArrayList<String> filenames = filepaths.collect(
        			Collector.of(
        					ArrayList<String>::new, 
        					(arrayList, element) -> {
        						if (element.toFile().isFile() && element.toFile().getParent().contains("cats")) {        							
        							arrayList.add(element.subpath(1, element.getNameCount()).toString());
        						}	
        					},
        					(left, right) -> {
        						left.addAll(right);
        						return left;
        					}
    					)
        			);
        	Path training_media = Paths.get("training_media");
        	ArrayList<Path> images = Main.collectImages(filenames, training_media);
        	normalized = Main.normalizeImages(
//        			images
        			new ArrayList<Path>(images.subList(0, DEBUG_COUNTER))
			);        	
        } catch (Exception exception) { exception.printStackTrace(); };
        
        if (normalized.isEmpty()) {
        	throw new InterruptedException("Failed to normalize BufferedImages");
        }
        
        System.out.printf("normalized size::{%d}\n", normalized.size());

		Layer[] network = {
//			Layer.Convolution2D(new int[]{256, 256, 32}, 3, 3, Activation::RELU),
			Layer.dropout(0.5, null),
			Layer.MaxPooling2D(new int[]{51, 51, 32}, 3, 5 /*should be default*/, 5, null),
//			Layer.Convolution2D(new int[]{51, 51, 64}, 3, 3, Activation::RELU),
			Layer.dropout(0.7, null),
			Layer.MaxPooling2D(new int[]{51, 51, 64}, 3, 5, 5, null),
			Layer.Flatten(),
			Layer.Dense(new int[]{1, 128}, Activation::Sigmoid),
			Layer.dropout(0.7, null),
			Layer.Dense(new int[]{128, 32}, Activation::RELU),
			Layer.dropout(0.7, null),
			Layer.Dense(new int[]{32, 2}, Activation::SoftMax)
		};

        Model model = Model.instantiate(network, Either.create_with(List.of(-1, 1)));
        model.compile_with(Optimizer.GradientDescent, Loss.MeanSquaredError.class);
        
        Either<Random, Integer> state = Either.create_with(42);
        ArrayList<BufferedImage>[] allocated = ModelUtils.allocate(normalized, 0.60, state, true, null);
        ArrayList<Mat> fortraining = allocated[0].stream().map(element -> toMatrix(element)).collect(Collectors.toCollection(ArrayList<Mat>::new));
        
//        allocated = ModelUtils.allocate(allocated[1], 0.5, state, true, null);
        Mat[] validation = new Mat[fortraining.size()];
        for (int i = 0; i < validation.length; ++i) {
        	validation[i] = Mat.zeros(fortraining.get(0).size(), fortraining.get(0).type());
        }
        
        Object displable = model.fit(fortraining.toArray(Mat[]::new), /*batch_size=*/ 25, /*epochs=*/ 25, /*verbose=*/ 1, /*validation=*/ validation, /*shuffle=*/ true);
////      TODO: display displayable to show progress
//        
//        BufferedImage image = ImageIO.read(new File(pathname));
//        Position<Float, Float> prediction = model.predict(image);
//        Debug.printo(prediction);
//        Main.display(image);
//        
//        model.save();
        Debug.print(fortraining.get(0).size());
        Debug.print(true);
    }
    
//  (MOVE:) -> [TestSuite]
    private static void test_of_forward_propogate_DENSE() {
    }
    
    public static String printable(Mat input) {
    	return input.dump().replace("\n", "");
    }
    
//  (MOVE:) -> [TestSuite]
    private static Mat[][] XOR_dataset() {
    	double[][] __input = {
	    		{0, 0},
	    		{0, 1},
	    		{1, 0},
	    		{1, 1}
	    };
	    Mat[] input = {
	    	new Mat(2, 1, CvType.CV_32FC1),
	    	new Mat(2, 1, CvType.CV_32FC1),
	    	new Mat(2, 1, CvType.CV_32FC1),
	    	new Mat(2, 1, CvType.CV_32FC1)
	    };
	    for (int i = 0; i < input.length; i++) {
			input[i].put(0, 0, __input[i]);
		}
	    
	    double[][] __expectedOutput = {
	    		{0},
	    		{1},
	    		{1},
	    		{0}
	    };
	    Mat[] expectedOutput = {
		    	new Mat(1, 1, CvType.CV_32FC1),
		    	new Mat(1, 1, CvType.CV_32FC1),
		    	new Mat(1, 1, CvType.CV_32FC1),
		    	new Mat(1, 1, CvType.CV_32FC1)
		    };
		    for (int i = 0; i < expectedOutput.length; i++) {
		    	expectedOutput[i].put(0, 0, __expectedOutput[i]);
			}
		return new Mat[][] {input, expectedOutput};
    }
    
//  (MOVE:) -> [TestSuite]
    private static void test_of_XOR() {
//    	https://replit.com/@AdimchimmaOdor/nptest
Boolean DEBUG = true;

		Mat[][] XOR_dataset = Main.XOR_dataset();
		Mat[] input = XOR_dataset[0];
		Mat[] expectedOutput = XOR_dataset[1];

	    Layer[] network = {
	    		Layer.Dense(new int[]{2, 3}, Activation::TANH),
//	    		Activation.TANH(),
	    		Layer.Dense(new int[]{3, 1}, Activation::TANH),
//	    		Activation.TANH()
	    };
	    
	    int epochs = 10000;
	    double learning_rate = 0.1;
	    
	    for (int e = 0; e < epochs; ++e) {
	    	double error = 0;
	    	for (int i = 0; i < input.length; ++i) {
//	    		(forward)
	    		Mat output = input[i];
	    		for (int j = 0; j < network.length; ++j) {
	    			output = network[j].forward_propogate_with(output);
	    		}

//	    		(error)
	    		error += Loss.MeanSquaredError.between(expectedOutput[i], output).val[0];

//	    		(backward)
	    		Mat gradient = Loss.MeanSquaredError.prime(expectedOutput[i], output);

	    		for (int j = network.length - 1; j >= 0; --j) {
	    			gradient = network[j].backward_propogate_with(gradient, learning_rate);
	    		}
	    	}
	    	error /= input.length;
	    	Debug.printc("%d/%d, error=%f", e + 1, epochs, error);
	    }
	    
Debug.prints("========================================================================");
	    for (int i = 0; i < input.length; ++i) {
	    	Mat output = input[i];
	    	for (int j = 0; j < network.length; ++j) {
	    		output = network[j].forward_propogate_with(output);
	    	}
	    	Debug.prints(output.dump());
	    	Debug.prints(expectedOutput[i].dump());
	    	System.out.println();
	    }
	}

//  (MOVE:) -> [TestSuite]
    private static void test_of_allocate() {
        ArrayList<Integer> labels = new ArrayList<>();
        labels.add(2);
        labels.add(2);
        labels.add(2);
        labels.add(3);
        labels.add(3);
        labels.add(3);
        labels.add(4);
        labels.add(4);
        labels.add(4);
        ArrayList<Integer> list = IntStream.rangeClosed(1, 9).boxed().collect(Collectors.toCollection(ArrayList::new));
        Debug.printo(list);
        ArrayList<?>[] result = ModelUtils.allocate(list, 0.75, ModelUtils.Either.create_with(12), true, labels);
        System.out.println(Arrays.toString(result));
        
        Debug.printo(list);
        list = IntStream.rangeClosed(1, 9).boxed().collect(Collectors.toCollection(ArrayList::new));
        result = ModelUtils.allocate(list, 0.75, ModelUtils.Either.create_with(12), true, null);
        System.out.println(Arrays.toString(result));
        
        Debug.printo(list);
        list = IntStream.rangeClosed(1, 9).boxed().collect(Collectors.toCollection(ArrayList::new));
        result = ModelUtils.allocate(list, 0.75, ModelUtils.Either.create_with(12), false, null);
        System.out.println(Arrays.toString(result));
	}

	public static void display(BufferedImage image) {
		try {
			Main.display_with(image, null, null);
		} catch (IOException e) {}
    }
	
	public static void display_with(BufferedImage image, Color color, String pathname) throws IOException {
		if (color == null) {
			color = Color.ORANGE;
		}
		
		Graphics2D graphics = (Graphics2D) image.getGraphics();
		graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

		if (pathname != null) {
			graphics.setStroke(new BasicStroke(3));
			graphics.setColor(color);
			
			if (!pathname.contains(".cat")) {
				pathname += ".cat";
			}

			BufferedReader reader = new BufferedReader(new FileReader(pathname));
			String line = reader.readLine(); // NOTE: should be only one line
			
			int[] annotations = Arrays.stream(line.strip().split(" ")).mapToInt(Integer::valueOf).toArray();
			for (int i = 1; i < annotations[0] * 2; i += 2) {
				
				int radius = 15;
				int x = annotations[i] - (radius / 2);
				int y = annotations[i + 1] - (radius / 2);
				
				graphics.drawOval(x, y, radius, radius);
			}
//			Debug.print(annotations);
		}

		JLabel label = new JLabel(new ImageIcon(image));
		JPanel panel = new JPanel();
		panel.add(label);

		JFrame frame = new JFrame();
		frame.setSize(new Dimension(image.getWidth(), image.getHeight()));
		frame.addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent WindowEvent) {
				System.exit(0);
			}
		});
		frame.add(panel);
		frame.setVisible(true);
	}
    
    public static BufferedImage toBufferedImage(Image image) {
    	if (image instanceof BufferedImage)
    		return (BufferedImage)image;
    	else {
    		BufferedImage bufferedImage = new BufferedImage(image.getHeight(null), image.getHeight(null), BufferedImage.TYPE_INT_ARGB);
    		
    		Graphics2D graphics = bufferedImage.createGraphics();
    		graphics.drawImage(image, 0, 0, null);
    		graphics.dispose();
    		
    		return bufferedImage;
    	}
    }
    
    public static Mat toMatrix(BufferedImage image) {
//      https://stackoverflow.com/questions/14958643/converting-bufferedimage-to-mat-in-opencv
      Mat matrix = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC3);
      byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
      matrix.put(0, 0, pixels);
      return matrix;
    }
    
    public static ArrayList<Path> collectImages(ArrayList<String> filenames, Path training_media) {
    	System.out.printf("extracting cat images from {%s}\n", training_media.toString());
    	
    	ArrayList<Path> paths = new ArrayList<Path>();
    	for (String filename : filenames) {
    		if (filename.endsWith(".cat") || filename.endsWith(".db"))
    			continue;
			else {
				paths.add(Paths.get(training_media.toAbsolutePath().toString(), filename));
			}
    	}
       	
    	return paths;
    }
    
    public static boolean confirm(BufferedImage image) {
    	return true; //TODO: actually check for failure
    }
    
    public static ArrayList<BufferedImage> normalizeImages(ArrayList<Path> paths) throws IOException, InterruptedException {
    	ArrayList<BufferedImage> normalized = new ArrayList<BufferedImage>();
    	normalized.ensureCapacity(paths.size());
    	
    	Incrementer index = new Incrementer();
    	Thread loading = new Thread(new ProgressBarRunnable(paths.size(), index));
    	loading.start();
    	for (int i = 0; i < paths.size(); ++i) {
    		Path path = paths.get(i);
    		synchronized(new int[] {i}) {
    			index.increment(1);
    		}
//System.out.printf("path::%s | normalized::%s\n", path, path.normalize().toString());
    		BufferedImage image = ImageIO.read(new File(path.normalize().toString()));
    		if (!Main.confirm(image)) {
    			continue;
    		}
    		
    		Mat manipulatable = Main.toMatrix(image);
//    		Imgproc.cvtColor(manipulatable, manipulatable, Imgproc.COLOR_BGR2GRAY);
    		Imgproc.cvtColor(manipulatable, manipulatable, Imgproc.COLOR_BGR2RGB);
    		Imgproc.resize(manipulatable, manipulatable, new Size(256, 256));
    		
    		MatOfByte buffer = new MatOfByte();
    		Imgcodecs.imencode(".jpg", manipulatable, buffer);
    		InputStream stream = new ByteArrayInputStream(buffer.toArray());
    		
    		normalized.add(ImageIO.read(stream));
    	}
    	loading.join();
//    	https://jenkov.com/tutorials/java-concurrency/creating-and-starting-threads.html
    	return normalized;
    }
    
}
