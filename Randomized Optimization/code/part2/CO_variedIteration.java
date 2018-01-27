package opt.test;

import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 *
 * @author Timothy Baba
 * @version 1.0
 */
public class CO_variedIteration {
    /** The n value (number of vertices)*/
    private static final int N = 80;
    private static DecimalFormat dfFormat = new DecimalFormat("0.000");

    public static void process() {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new CountOnesEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        deleteFiles();


        int[] iterations = new int[51];
        iterations[0] = 0;
        for (int i =  1; i < 51; i++ ) {
            iterations[i] = iterations[i -1] + 20;
        }

        for (int itr: iterations) {
            FileWriter RHC_fitness = null;
            try {
                RHC_fitness = new FileWriter("./src/opt/test/optData/CO_variedIteration/RHC.txt", true);
            } catch (Exception e) {
                System.out.println("did not create file");
                System.exit(0);
            }

            FileWriter SA_fitness = null;
            try {
                SA_fitness = new FileWriter("./src/opt/test/optData/CO_variedIteration/SA.txt", true);
            } catch (Exception e) {
                System.out.println("did not create file");
                System.exit(0);
            }

            FileWriter GA_fitness = null;
            try {
                GA_fitness = new FileWriter("./src/opt/test/optData/CO_variedIteration/GA.txt", true);
            } catch (Exception e) {
                System.out.println("did not create file");
                System.exit(0);
            }
            System.out.println("Processing CO_variedIteration itration: " + itr);

            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit_rhc = new FixedIterationTrainer(rhc, 200);
            fit_rhc.train();
            //System.out.println(ef.value(rhc.getOptimal()));

            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
            FixedIterationTrainer fit_sa = new FixedIterationTrainer(sa, 200);
            fit_sa.train();
            //System.out.println(ef.value(sa.getOptimal()));

            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 20, 0, gap);
            FixedIterationTrainer fit_ga = new FixedIterationTrainer(ga, 300);
            fit_ga.train();


            for (int i = 0; i < itr; i++) {
                fit_rhc.train();
                //System.out.println(ef.value(rhc.getOptimal()));
                fit_sa.train();
                //System.out.println(ef.value(sa.getOptimal()));

                fit_ga.train();
                //System.out.println(ef.value(ga.getOptimal()));
            }
            try {
                RHC_fitness.write(dfFormat.format(ef.value(rhc.getOptimal())));
                RHC_fitness.write("\n");
                RHC_fitness.close();
            } catch (Exception e) {
                System.out.println("Did not write to file");
                System.exit(0);
            }
            try {
                SA_fitness.write(dfFormat.format(ef.value(sa.getOptimal())));
                SA_fitness.write("\n");
                SA_fitness.close();
            } catch (Exception e) {
                System.out.println("Did not write to file");
                System.exit(0);
            }
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
            File file =  new File("./src/opt/test/optData/Co_variedIteration/" + fileName);
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