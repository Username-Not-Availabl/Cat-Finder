package com.csc.honors_module;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import com.csc.honors_module.MathUtils.Activation;
import com.csc.honors_module.MathUtils.Loss;


public class Main extends ClassLoader {
    public static void main(String[] arguments) throws IOException, InterruptedException, ClassNotFoundException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
         
//        String pathname = "training_media\\archive\\CAT_00\\00000001_000.jpg";
        String pathname = "archive\\CAT_00\\00000001_000.jpg";
        

//        test_of_allocate();
//        test_of_forward_propogate_DENSE();
        test_of_XOR();
        
        System.out.print(true);
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
//        
//        ArrayList<BufferedImage> normalized = new ArrayList<BufferedImage>();
//        
////        TODO: make Main.collect images take Stream<Path> so all this work will be done inside
//        try (Stream<Path> filepaths = Files.walk(Paths.get("training_media/archive"))) {
//        	ArrayList<String> filenames = filepaths.collect(
//        			Collector.of(
//        					ArrayList<String>::new, 
//        					(arrayList, element) -> {
//        						if (element.toFile().isFile()) {        							
//        							arrayList.add(element.subpath(1, element.getNameCount()).toString());
//        						}	
//        					},
//        					(left, right) -> {
//				        		left.addAll(right);
//			 	        		return left;
//        					}
//        			)
//        	);
//        	Path training_media = Paths.get("training_media");
//        	ArrayList<Path> images = Main.collectImages(filenames, training_media);
//        	normalized = Main.normalizeImages(images);        	
//        } catch (Exception exception) { exception.printStackTrace(); };
//        
//        if (normalized.isEmpty()) {
//        	throw new InterruptedException("Failed to normalize BufferedImages");
//        }
//        
//        System.out.println(normalized.size());
//        
//        Layer[] layers = new Layer[] {
//				Layer.Input(), //TODO: have input layer be called implicitly
//				Layer.Convolution2D(),
//				Layer.dropout(0.5),
//				Layer.MaxPooling2D(),
//				Layer.Convolution2D(),
//				Layer.dropout(0.7),
//				Layer.MaxPooling2D(),
//				Layer.Flatten(),
//				Layer.Dense(),
//				Layer.dropout(0.7),
//				Layer.Dense(),
//				Layer.dropout(0.7),
//				Layer.Dense()
//		};
//        Model model = Model.initiate(layers, Either.create_with(List.of(-1, 1)));
//        model.compile();
//        
//        Either<Random, Integer> state = Either.create_with(42);
//        ArrayList<?>[] allocated = ModelUtils.allocate(normalized, 0.60, state, true, null);
//        ArrayList<?> fortraining = allocated[0];
//        
//        allocated = ModelUtils.allocate(allocated[1], 0.5, state, true, null);
//        Object displable = model.fit(fortraining, /*batch_size=*/ 25, /*epochs=*/ 25, /*verbose=*/ 1, /*validation=*/ allocated[1], /*shuffle=*/ true);
////      TODO: display displayable to show progress
//        
//        BufferedImage image = ImageIO.read(new File(pathname));
//        Position<Float, Float> prediction = model.predict(image);
//        Debug.printo(prediction);
//        Main.display(image);
//        
//        model.save();
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
        Graphics2D graphics = (Graphics2D) image.getGraphics();
        graphics.setStroke(new BasicStroke(3));
        graphics.setColor(Color.BLUE);
        graphics.drawRect(10, 10, image.getWidth() - 20, image.getHeight() - 20);

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
    	return true;
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
System.out.printf("path::%s | normalized::%s\n", path, path.normalize().toString());
    		BufferedImage image = ImageIO.read(new File(path.normalize().toString()));
    		if (!Main.confirm(image)) {
    			continue;
    		}
    		
    		Mat manipulatable = Main.toMatrix(image);
    		Imgproc.cvtColor(manipulatable, manipulatable, Imgproc.COLOR_BGR2GRAY);
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
