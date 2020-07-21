import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.function.UnaryOperator;

public class Main {

    public static void main(String[] args) throws IOException {
        InputStreamReader r = new InputStreamReader(System.in);
        BufferedReader br = new BufferedReader(r);
        System.out.println("Hello, there!");
        System.out.println("[0] - get trainned NN \n[1] - train NN");
        int choice = 1;
        try {
            choice = Integer.parseInt(br.readLine());
        } catch (Exception e) {
            System.out.println("Incorrect format!");
        }

        if (choice == 0) {
            getTrainedNN();
        } else if (choice == 1) {
            digits();
        } else {
            System.out.println("Incorrect input. Try again later!k");
        }
    }


    private static void digits() throws IOException {
        UnaryOperator<Double> sigmoid = x -> 1 / (1 + Math.exp(-x));
        UnaryOperator<Double> dsigmoid = y -> y * (1 - y);
        NeuralNetwork nNet = new NeuralNetwork(0.001, sigmoid, dsigmoid, 784, 512, 128, 32, 10);

        int samples = 60000;
        BufferedImage[] images = new BufferedImage[samples];
        int[] digits = new int[samples];
        File[] imagesFiles = new File("C:/Users/Asus/Desktop/NEURAL_NETWORKS/digit_recognition/train").listFiles();
        for (int i = 0; i < samples; i++) {
            images[i] = ImageIO.read(imagesFiles[i]);
            digits[i] = Integer.parseInt(imagesFiles[i].getName().charAt(10) + "");
        }

        double[][] inputs = new double[samples][784];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < 28; j++) {
                for (int k = 0; k < 28; k++) {
                    inputs[i][j + k * 28] = (images[i].getRGB(j, k) & 0xff) / 255.0;
                }
            }
        }

        int epochs = 1000;
        for (int i = 1; i < epochs; i++) {
            int correct = 0;
            double errorSum = 0;
            int batchSize = 10000;
            for (int j = 0; j < batchSize; j++) {
                int imgIndex = (int) (Math.random() * samples);
                double[] targets = new double[10];
                int digit = digits[imgIndex];
                targets[digit] = 1;

                double[] outputs = nNet.feedForward(inputs[imgIndex]);
                int maxDigit = 0;
                double maxDigitWeight = -1;
                for (int k = 0; k < 10; k++) {
                    if (outputs[k] > maxDigitWeight) {
                        maxDigitWeight = outputs[k];
                        maxDigit = k;
                    }
                }
                if (digit == maxDigit) correct++;
                for (int k = 0; k < 10; k++) {
                    errorSum += (targets[k] - outputs[k]) * (targets[k] - outputs[k]);
                }
                nNet.backpropagation(targets);
            }
            System.out.println("epoch: " + i + ". correct: " + correct + ". error: " + errorSum);
        }

        FormDigits f = new FormDigits(nNet);
        new Thread(f).start();
        savParamsNN(nNet);
    }

    private static void savParamsNN(NeuralNetwork nNet) {
        try {
            ObjectOutputStream objectOutputStream = new ObjectOutputStream(
                    new FileOutputStream("TrainedNN.out"));
            objectOutputStream.writeObject(nNet);
            objectOutputStream.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void getTrainedNN() {
        try {
            ObjectInputStream objectInputStream = new ObjectInputStream(
                    new FileInputStream("TrainedNN.out"));
            NeuralNetwork nNet = (NeuralNetwork) objectInputStream.readObject();
            objectInputStream.close();
            UnaryOperator<Double> sigmoid = x -> 1 / (1 + Math.exp(-x));
            UnaryOperator<Double> dsigmoid = y -> y * (1 - y);
            nNet.setActivation(sigmoid);
            nNet.setDerivative(dsigmoid);
            FormDigits f = new FormDigits(nNet);
            new Thread(f).start();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}