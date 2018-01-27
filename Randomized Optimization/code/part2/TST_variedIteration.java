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
public class TST_variedIteration {
    /** The n value */
    private static final int N = 50;
    private static DecimalFormat df = new DecimalFormat("0.000");
    /**
     * The test main
     *
     */
    public static void process() {
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        deleteFiles();

        int[] iterations = new int[51];
        iterations[0] = 0;
        for (int i =  1; i < 51; i++ ) {
            iterations[i] = iterations[i -1] + 20;
        }

        for (int itr: iterations) {
            FileWriter RHC_fitness = null;
            try {
                RHC_fitness = new FileWriter("./src/opt/test/optData/TST_variedIteration/RHC.txt", true);
            } catch (Exception e) {
                System.out.println("did not create file");
                System.exit(0);
            }

            FileWriter SA_fitness = null;
            try {
                SA_fitness = new FileWriter("./src/opt/test/optData/TST_variedIteration/SA.txt", true);
            } catch (Exception e) {
                System.out.println("did not create file");
                System.exit(0);
            }

            FileWriter GA_fitness = null;
            try {
                GA_fitness = new FileWriter("./src/opt/test/optData/TST_variedIteration/GA.txt", true);
            } catch (Exception e) {
                System.out.println("did not create file");
                System.exit(0);
            }

            System.out.println("Processing TST_variedIteration itration: " + itr);
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit_rhc = new FixedIterationTrainer(rhc, 200000);

            SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
            FixedIterationTrainer fit_sa = new FixedIterationTrainer(sa, 200000);

            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
            FixedIterationTrainer fit_ga = new FixedIterationTrainer(ga, 1000);

            for (int i = 0; i < itr; i++) {
                fit_rhc.train();
                //System.out.println(ef.value(rhc.getOptimal()));

                fit_sa.train();
                //System.out.println(ef.value(sa.getOptimal()));

                fit_ga.train();
                //System.out.println(ef.value(ga.getOptimal()));
            }
            try {
                RHC_fitness.write(df.format(ef.value(rhc.getOptimal())));
                RHC_fitness.write("\n");
                RHC_fitness.close();
            } catch (Exception e) {
                System.out.println("Did not write to file");
                System.exit(0);
            }
            try {
                SA_fitness.write(df.format(ef.value(sa.getOptimal())));
                SA_fitness.write("\n");
                SA_fitness.close();
            } catch (Exception e) {
                System.out.println("Did not write to file");
                System.exit(0);
            }
            try {
                GA_fitness.write(df.format(ef.value(ga.getOptimal())));
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
            File  file =  new File("./src/opt/test/optData/TST_variedIteration/" + fileName);
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
