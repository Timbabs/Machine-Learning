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
public class CO_varied_N {
    /** The n value (number of vertices)*/
    private static int N;// = 80;
    private static DecimalFormat dfFormat = new DecimalFormat("0.000");
    public static void process(Integer[] args) {
        deleteFiles();
        for (Integer n: args) {
            N = n;
            System.out.println("processing CO_varied_N N = " + N);
            int[] ranges = new int[N];
            Arrays.fill(ranges, 2);

            FileWriter RHC_fitness = null;
            try {
                RHC_fitness = new FileWriter("./src/opt/test/optData/CO_varied_N/RHC.txt", true);
            } catch (Exception e) {
                System.out.println("did not create file");
                System.exit(0);
            }

            FileWriter SA_fitness = null;
            try {
                SA_fitness = new FileWriter("./src/opt/test/optData/CO_varied_N/SA.txt", true);
            } catch (Exception e) {
                System.out.println("did not create file");
                System.exit(0);
            }

            FileWriter GA_fitness = null;
            try {
                GA_fitness = new FileWriter("./src/opt/test/optData/CO_varied_N/GA.txt", true);
            } catch (Exception e) {
                System.out.println("did not create file");
                System.exit(0);
            }


            EvaluationFunction ef = new CountOnesEvaluationFunction();
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new UniformCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200);
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

            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
            fit = new FixedIterationTrainer(sa, 200);
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

            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 20, 0, gap);
            fit = new FixedIterationTrainer(ga, 300);
            fit.train();
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
            File file =  new File("./src/opt/test/optData/CO_varied_N/" + fileName);
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