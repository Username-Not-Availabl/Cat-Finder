package com.csc.honors_module;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.BasicStroke;
import java.awt.Graphics2D;
import java.awt.event.WindowEvent;
import java.awt.event.WindowAdapter;
import java.awt.image.BufferedImage;

import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

// import com.csc.honors_module.opencv.core.Core;

public class Main {
    public static void main(String[] arguments) throws IOException {
        // System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String pathname = "com\\csc\\honors_module\\archive\\CAT_00\\00000001_000.jpg";
        BufferedImage image = ImageIO.read(new File(pathname));

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
}
