package opt.test.myTests;
import opt.test.CountOnesTest;
import opt.test.FlipFlopTest;
import opt.test.TravelingSalesmanTest;

/**
 * Created by timothybaba on 11/6/17.
 */
public class Tester {
    public static void main(String[] args) {
        Integer[] N = {25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500};
        TravelingSalesmanTest.process(N);
        FlipFlopTest.process(N);
        CountOnesTest.process(N);
    }
}
