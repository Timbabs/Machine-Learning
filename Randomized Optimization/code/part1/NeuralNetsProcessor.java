import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer
 * or more than 15 rings.
 *
 * @author Timothy Baba
 * @version 1.0
 */
public class NeuralNetsProcessor {
    private static Instance[] trainData_Instances;
    private static Instance[] testData_Instances;

    private static int inputLayer = 30, /*hiddenLayer = 5,*/ outputLayer = 1, trainingIterations;// = 1000;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet trainDataSet;

    private static String[] oaNames;//{"RHC", "SA", "GA"};
    private static BackPropagationNetwork[] networks;
    private static NeuralNetworkOptimizationProblem[] nnop;

    private static OptimizationAlgorithm[] oa;

    private static DecimalFormat df = new DecimalFormat("0.000");
    private static HashMap<String, LinkedList<Double>> algoParameters;

    public NeuralNetsProcessor(HashMap<String, LinkedList<Double>> algoParameters) {

        NeuralNetsProcessor.algoParameters = algoParameters;
        oaNames = new String[algoParameters.size()];
        int i = 0;
        for (String s: algoParameters.keySet()) {
            oaNames[i] = s;
            i++;
        }
        networks =  new BackPropagationNetwork[algoParameters.size()];
        nnop = new NeuralNetworkOptimizationProblem[algoParameters.size()];
        oa = new OptimizationAlgorithm[algoParameters.size()];
    }


    public static void initializeAlgorithms() {
        for (int i = 0; i < oaNames.length; i++) {
            switch (oaNames[i]) {
                case "RHC": oa[i] = new RandomizedHillClimbing(nnop[i]);
                    break;
                case "SA":
                    double startTemp = algoParameters.get("SA").get(0);
                    double coolingExponent = algoParameters.get("SA").get(1);
                    oa[i] = new SimulatedAnnealing(startTemp, coolingExponent, nnop[i]);
                    break;
                case "GA":
                    int populationSize = (algoParameters.get("GA").get(0)).intValue();
                    int toMate = (algoParameters.get("GA").get(1)).intValue();
                    int toMutate = (algoParameters.get("GA").get(2)).intValue();
                    oa[i] = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, nnop[i]);
                    break;


            }
        }
    }

    public static HashMap<String, Double> process(int iterations, DataSet trainDataSet, Instance[] train_data, Instance[] test_data, String variedParameter) {
        HashMap<String, Double> map = new HashMap<>();
        System.out.print("processing.....[ [Experiment: ");
        String varied = "";
        switch (variedParameter) {
            case "iteration":
                varied = "variedIteration";
                System.out.print(varied+ "] [iterations: " + iterations + "] ]......");
                break;
            case "trainSize":
                varied = "variedTrainSize";
                System.out.print(varied+ "] [trainSize: " + train_data.length + "] ]......");
                break;
            case "SA/CoolE":
                varied = "variedCoolingExponent";
                System.out.print(varied+ "] ]......");
                break;
            case "SA/Temp":
                varied = "variedTemp";
                System.out.print(varied+ "] ]......");
                break;
            case "GA/Mutate":
                varied = "variedMutate";
                System.out.print(varied+ "] ]......");
                break;
            case "GA/Mate":
                varied = "variedMate";
                System.out.print(varied+ "] ]......");
                break;
            case "GA/size":
                varied = "variedPopulation";
                System.out.print(varied+ "] ]......");
                break;
        }

        System.out.println();
        NeuralNetsProcessor.trainDataSet = trainDataSet;
        NeuralNetsProcessor.trainData_Instances= train_data;
        NeuralNetsProcessor.testData_Instances = test_data;
        NeuralNetsProcessor.trainingIterations = iterations;

        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, 30, 30, 30, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(trainDataSet, networks[i], measure);
        }

        initializeAlgorithms();

        for(int i = 0; i < oa.length; i++) {
            FileWriter outputFileTrainAccuracy = null;
            try {
                outputFileTrainAccuracy = new FileWriter("./src/opt/test/myTests/outputData/"+varied+"/" + oaNames[i]+ "/" + oaNames[i]+"_trainAccuracy_70.txt", true);
            } catch (Exception e) {
                System.out.println("did not create file why");
                System.exit(0);
            }

            FileWriter outputFileTestAccuracy = null;
            try {
                outputFileTestAccuracy = new FileWriter("./src/opt/test/myTests/outputData/"+varied+"/" + oaNames[i]+ "/"+ oaNames[i]+"_testAccuracy_70.txt", true);
            } catch (Exception e) {
                System.out.println("did not create file");
                System.exit(0);
            }

            // Train the model
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9); //time in seconds
            map.put(oaNames[i], trainingTime/iterations);


            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            //Test on train set
            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < trainData_Instances.length; j++) {
                networks[i].setInputValues(trainData_Instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(trainData_Instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9); //time in seconds

            double accuracy = correct/(correct+incorrect)*100;

            try {
                outputFileTrainAccuracy.write(df.format(accuracy));
                outputFileTrainAccuracy.write("\n");
                outputFileTrainAccuracy.close();
            } catch (Exception e) {
                System.out.println("did not write to file");

            }
            //Test on test set

            start = System.nanoTime(); correct = 0; incorrect = 0;

            for(int j = 0; j < testData_Instances.length; j++) {
                networks[i].setInputValues(testData_Instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(testData_Instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            accuracy = correct/(correct+incorrect)*100;

            try {
                outputFileTestAccuracy.write(df.format(accuracy));
                outputFileTestAccuracy.write("\n");
                outputFileTestAccuracy.close();
            } catch (Exception e) {
                System.out.println("did not write to file");

            }
        }
        return map;

    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
       // System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < trainData_Instances.length; j++) {
                network.setInputValues(trainData_Instances[j].getData());
                network.run();

                Instance output = trainData_Instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

        }
    }

}
