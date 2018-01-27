import shared.*;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;

/**
 * Created by timothybaba on 11/2/17.
 */
public class NeuralNetsDriver {
    public static void main(String[] args) {
        HashMap<String, LinkedList<Double>> map = new HashMap<>();
        map.put("RHC", new LinkedList<>());
        map.put("SA",new LinkedList<>(Arrays.asList(1E11, .95)));
        map.put("GA", new LinkedList<>(Arrays.asList(200.0, 100.0, 10.0)));
        new NeuralNetsProcessor(map);
        deleteFiles();
        variedIterationTest();
        variedTrainingSetTest();
        variedHyperParameters();
        optimalParam();

        int[] iterations = new int[51];
        iterations[0] = 0;
        for (int i =  1; i < 51; i++ ) {
            iterations[i] = iterations[i -1] + 20;
        }


    }
    //optimal prunning
    public static void optimalParam() {
        int[] iterations = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};

        String cancer_trainData_path = "./src/opt/test/myTests/inputData/train_70.csv";
        int cancer_trainData_length = 398;
        int no_of_attribures = 30;

        String cancer_testData_path = "./src/opt/test/myTests/inputData/test_30.csv";
        int cancer_testData_length = 171;

        Instance[] train_data = DataParser.getData(cancer_trainData_path,cancer_trainData_length,no_of_attribures);
        Instance[] test_data = DataParser.getData(cancer_testData_path, cancer_testData_length,no_of_attribures);
        DataSet train_set = new DataSet(train_data);


        HashMap<String, LinkedList<Double>> map = new HashMap<>();
        map.put("RHC", new LinkedList<>());
        map.put("SA",new LinkedList<>(Arrays.asList(1E13, .35)));
        map.put("GA", new LinkedList<>(Arrays.asList(200.0, 100.0, 75.0)));
        new NeuralNetsProcessor(map);

        double RHC_time = 0.0;
        double SA_Time = 0.0;
        double GA_Time = 0.0;
        HashMap<String, Double> time_result = new HashMap<>();
        for (int itr:  iterations) {

            time_result = NeuralNetsProcessor.process(itr, train_set, train_data, test_data, "iteration");
            SA_Time += time_result.get("SA");
            GA_Time += time_result.get("GA");
            RHC_time += time_result.get("RHC");
        }

        System.out.println("\nAverage training time for RHC is : " + RHC_time / iterations.length);
        System.out.println("\nAverage training time for SA is : " + SA_Time / iterations.length);
        System.out.println("\nAverage training time for GA is : " + GA_Time / iterations.length);


    }
    //SA &GA param pruning
    public static void variedHyperParameters() {
        double[] coolingExponent = {0.2, 0.35, 0.5, 0.65, 0.8, 0.95};
        double[] startTemp = {1E9, 1E10, 1E11, 1E12, 1E13};
        double[] population = {100, 200, 300, 400};
        double[] mate = {25, 50, 75, 100};
        double[] mutate = {0, 25, 50, 75, 100};

        String diabetes_trainData_path = "./src/opt/test/myTests/inputData/train_70.csv";
        int diabetes_trainData_length = 398;
        int no_of_attribures = 30;

        String diabetes_testData_path = "./src/opt/test/myTests/inputData/test_30.csv";
        int diabetes_testData_length = 171;

        Instance[] train_data = DataParser.getData(diabetes_trainData_path,diabetes_trainData_length,no_of_attribures);
        Instance[] test_data = DataParser.getData(diabetes_testData_path, diabetes_testData_length,no_of_attribures);
        DataSet train_set = new DataSet(train_data);

        HashMap<String, LinkedList<Double>> map;
        int itr = 10;
        for (double CE: coolingExponent) {
            map = new HashMap<>();
            map.put("SA",new LinkedList<>(Arrays.asList(1E11, CE)));
            new NeuralNetsProcessor(map);
            NeuralNetsProcessor.process(itr, train_set, train_data, test_data, "SA/CoolE");
        }
        for (double T: startTemp) {
            map = new HashMap<>();
            map.put("SA",new LinkedList<>(Arrays.asList(T, 0.95)));
            new NeuralNetsProcessor(map);
            NeuralNetsProcessor.process(itr, train_set, train_data, test_data, "SA/Temp");
        }

        for (double p: population) {
            map = new HashMap<>();
            map.put("GA",new LinkedList<>(Arrays.asList(p, 100.0, 10.0)));
            new NeuralNetsProcessor(map);
            NeuralNetsProcessor.process(itr, train_set, train_data, test_data, "GA/size");
        }

        for (double ma: mate) {
            map = new HashMap<>();
            map.put("GA",new LinkedList<>(Arrays.asList(200.0, ma, 10.0)));
            new NeuralNetsProcessor(map);
            NeuralNetsProcessor.process(itr, train_set, train_data, test_data, "GA/Mate");
        }
        for (double mu: mutate) {
            map = new HashMap<>();
            map.put("GA",new LinkedList<>(Arrays.asList(200.0, 100.0, mu)));
            new NeuralNetsProcessor(map);
            NeuralNetsProcessor.process(itr, train_set, train_data, test_data, "GA/Mutate");
        }
    }

    public static void variedIterationTest() {
        //int[] iterations = new int[]{10, 20, 30, 40, 50, 60};
        int[] iterations = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};

        String diabetes_trainData_path = "./src/opt/test/myTests/inputData/train_70.csv";
        int diabetes_trainData_length = 398;
        int no_of_attribures = 30;

        String diabetes_testData_path = "./src/opt/test/myTests/inputData/test_30.csv";
        int diabetes_testData_length = 171;

        Instance[] train_data = DataParser.getData(diabetes_trainData_path,diabetes_trainData_length,no_of_attribures);
        Instance[] test_data = DataParser.getData(diabetes_testData_path, diabetes_testData_length,no_of_attribures);
        DataSet train_set = new DataSet(train_data);

       for (int itr:  iterations) {
           NeuralNetsProcessor.process(itr, train_set, train_data, test_data, "iteration");
       }
    }

    public static void variedTrainingSetTest() {
        int[] splits = new int[]{10, 20,30,40,50,60,70,80, 90};
        int[] cancer_train_lengths = new int[]{39, 79, 119, 159, 199, 239, 279, 319, 359};
        int no_of_attribures = 30;

        String cancer_testData_path = "./src/opt/test/myTests/inputData/test_30.csv";
        int diabetes_testData_length = 171;
        Instance[] test_data = DataParser.getData(cancer_testData_path, diabetes_testData_length,no_of_attribures);

        int iteration = 100;
        for (int i = 0; i < splits.length; i++) {
            int split = splits[i];
            String diabetes_trainData_path = "./src/opt/test/myTests/inputData/train_70_"+split+".csv";

            Instance[] train_data = DataParser.getData(diabetes_trainData_path,cancer_train_lengths[i],no_of_attribures);
            DataSet train_set = new DataSet(train_data);
            NeuralNetsProcessor.process(iteration, train_set, train_data, test_data, "trainSize");
        }

    }

    private static void deleteFiles() {
        String[] varied = {"variedCoolingExponent", "variedIteration", "variedMate",
                "variedMutate", "variedPopulation", "variedTemp", "variedTrainSize"};
        String[] dataset = {"test", "train"};
        String[]algorithms = {"RHC", "SA", "GA"};
        for (String var:varied) {
            for (String algo: algorithms) {
                for(String data : dataset) {
                    String fileName = algo + "_" + data + "Accuracy_70.txt";
                    File file = new File("./src/opt/test/myTests/outputData/" + var + "/" + algo + "/" + fileName);
                    System.out.print(file);
                    if (file.exists()) {
                        if(file.delete())
                        {
                            System.out.print(fileName + " deleted successfully");
                            System.out.println();
                        }
                        else
                        {
                            System.out.println("Failed to delete " + fileName);
                        }
                    } else {
                        System.out.println("File does not exist.");
                    }

                }
            }
        }
        System.out.println("\n\tFINISHED FILE DELETION");
    }
}
