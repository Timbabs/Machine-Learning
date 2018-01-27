package opt.test;

import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.Random;

import dist.DiscretePermutationDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import shared.FixedIterationTrainer;

/**
 *
 * @author Timothy Baba
 * @version 1.0
 */
public class TST_varied_N{
    /** The n value */
    private static  int N; //= 50;
    private static DecimalFormat dfFormat = new DecimalFormat("0.000");
    /**
     * The test main
     * @param args ignored
     */
    public static void process(Integer[] args) {
        deleteFiles();
        for (Integer n : args) {
            N = n;
            System.out.println("processing TST_varied_N N = " + N);
            Random random = new Random();
            // create the random points
            double[][] points = new double[N][2];
            for (int i = 0; i < points.length; i++) {
                points[i][0] = random.nextDouble();
                points[i][1] = random.nextDouble();
            }

            FileWriter RHC_fitness = null;
            try {
                RHC_fitness = new FileWriter("./src/opt/test/optData/TST_varied_N/RHC.txt", true);
            } catch (Exception e) {
                System.out.println("did not create file");
                System.exit(0);
            }

            FileWriter SA_fitness = null;
            try {
                SA_fitness = new FileWriter("./src/opt/test/optData/TST_varied_N/SA.txt", true);
            } catch (Exception e) {
                System.out.println("did not create file");
                System.exit(0);
            }

            FileWriter GA_fitness = null;
            try {
                GA_fitness = new FileWriter("./src/opt/test/optData/TST_varied_N/GA.txt", true);
            } catch (Exception e) {
                System.out.println("did not create file");
                System.exit(0);
            }

            // for rhc, sa, and ga we use a permutation based encoding
            TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
            Distribution odd = new DiscretePermutationDistribution(N);
            NeighborFunction nf = new SwapNeighbor();
            MutationFunction mf = new SwapMutation();
            CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
            fit.train();
            // System.out.println(ef.value(rhc.getOptimal()));
            try {
                RHC_fitness.write(dfFormat.format(ef.value(rhc.getOptimal())));
                RHC_fitness.write("\n");
                RHC_fitness.close();
            } catch (Exception e) {
                System.out.println("Did not write to file");
                System.exit(0);
            }

            SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
            fit = new FixedIterationTrainer(sa, 200000);
            fit.train();
            //System.out.println(ef.value(sa.getOptimal()));
            try {
                SA_fitness.write(dfFormat.format(ef.value(sa.getOptimal())));
                SA_fitness.write("\n");
                SA_fitness.close();
            } catch (Exception e) {
                System.out.println("Did not write to file");
                System.exit(0);
            }

            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
            fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
            //System.out.println(ef.value(ga.getOptimal()));
            try {
                GA_fitness.write(dfFormat.format(ef.value(ga.getOptimal())));
                GA_fitness.write("\n");
                GA_fitness.close();
            } catch (Exception e) {
                System.out.println("Did not write to file");
                System.exit(0);
            }

        }
    }

    private static void deleteFiles() {
        String[] algorithms = {"RHC", "SA", "GA"};

        for(String algo: algorithms) {
            String fileName = algo + ".txt";
            File file =  new File("./src/opt/test/optData/TST_varied_N/" + fileName);
            if (file.exists()) {
                if(file.delete())
                {
                    System.out.println(fileName + " deleted successfully");
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