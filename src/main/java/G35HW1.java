import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import scala.Tuple2;
import java.util.ArrayList;
import java.util.Iterator;

public class G35HW1 {

    //computes the squared Euclidean distance between two points
    public static double distance(Vector point1, Vector point2) {
        double distance = Vectors.sqdist(point1, point2); //no sqrt, more optimization
        return distance;
    }

    public static ArrayList<Tuple2<Vector, String>> FairFFT (ArrayList<Tuple2<Vector, String>> U, int ka, int kb) {
        int countA = 0;
        int countB = 0;

        ArrayList<Tuple2<Vector, String>> outputList = new ArrayList<Tuple2<Vector, String>>();

        int firstIndex = -1;  //index of the first valid center
        boolean validA;     //true if point is from A and countA < ka
        boolean validB;     //true if point is from B and countB < kb
        boolean[] inS = new boolean[U.size()]; //track which points are already centers

        //start find first valid center respecting ka and kb
        for(int i = 0; i < U.size(); i++){
            validA = U.get(i)._2().equals("A") && countA < ka;
            validB = U.get(i)._2().equals("B") && countB < kb;
            if(validA || validB){
                firstIndex = i;
                inS[i] = true;
                break;
            }
        }
        outputList.add(U.get(firstIndex));

        if (U.get(firstIndex)._2.equals("A")){
            countA++;
        } else{
            countB++;
        }

        double[] minDist = new double[U.size()];
        //set minDist[i] the distance between first center and point i
        for (int i = 0; i < U.size(); i++) {
            minDist[i] = distance(outputList.get(0)._1(), U.get(i)._1());
        }

        int bestIndex = -1;   //index of the farthest point found so far
        double bigDist = -1;  //distance of the farthest point found so far

        //repeat until we have exactly ka centers from A and kb centers from B
        while(countA < ka || countB < kb){

            //find the point with the largest minimum distance to current centers
            for(int i = 0; i < U.size(); i++) {
                if (!inS[i]) {  //only if point i is not already center
                    //only consider points whose group still has remaining budget
                    validA = U.get(i)._2().equals("A") && countA < ka;
                    validB = U.get(i)._2().equals("B") && countB < kb;

                    if (minDist[i] > bigDist && (validA || validB)) {
                        bigDist = minDist[i];
                        bestIndex = i;
                    }
                }
            }

            //add the farthest valid point as a new center
            outputList.add(U.get(bestIndex));
            //set the farthest valid point as center
            inS[bestIndex] = true;
            //increase counters
            if(U.get(bestIndex)._2.equals("A")){
                countA++;
            } else{
                countB++;
            }

            //update minDist considering the new center just added
            for (int i = 0; i < U.size(); i++) {
                if(!inS[i]) { //only if point i is not already center
                    double d = distance(U.get(bestIndex)._1(), U.get(i)._1());
                    if (d < minDist[i]) {
                        minDist[i] = d;
                    }
                }
            }
            //reset for next iteration
            bestIndex = -1;
            bigDist = -1;
        }
        return outputList;
    }

    public static Iterator<Tuple2<Vector, String>> applyFairFFTOnPartition (Iterator<Tuple2<Vector,String>> partition, int ka, int kb) {

        //convert partition iterator to ArrayList for FairFFT
        ArrayList<Tuple2<Vector, String>> partitionList = new ArrayList<>();

        //counter for check effective A's and B's points
        int counterA = 0;
        int counterB = 0;
        //convert Iterator to Arraylist and count A's and B's points
        while (partition.hasNext()) {
            Tuple2<Vector, String> p = partition.next();
            partitionList.add(p);
            if (p._2().equals("A")){
                counterA++;
            }
            else{
                counterB++;
            }
        }

        //don't ask for more centers than available points per group
        int effectiveKa = Math.min(ka, counterA);
        int effectiveKb = Math.min(kb, counterB);

        //apply FairFFT on this partition and return centers as iterator
        ArrayList<Tuple2<Vector, String>> centersR1 = FairFFT(partitionList, effectiveKa, effectiveKb);
        return centersR1.iterator();
    }

    public static ArrayList<Tuple2<Vector, String>> MRFairFFT (JavaRDD<Tuple2<Vector,String>> U, int ka, int kb, int L) {

        //round1 apply FairFFT on each partition in parallel, ka*L and kb*L for better accuracy
        JavaPairRDD<Vector, String> outputR1 = U.mapPartitionsToPair(partition -> applyFairFFTOnPartition(partition, ka*L, kb*L));

        //convert all partition centers into a single ArrayList for use FairFFT method
        ArrayList<Tuple2<Vector, String>> outputR1List = new ArrayList<Tuple2<Vector, String>>();
        outputR1List = new ArrayList<>(outputR1.collect());

        //round2 apply FairFFT on the coreset to get final centers
        ArrayList<Tuple2<Vector, String>> centersR2 = FairFFT(outputR1List, ka, kb);
        return centersR2;
    }

    public static void main(String[] args) throws Exception {
        //check the correct number of arguments
        if (args.length != 4) {
            throw new IllegalArgumentException("Error: expected 4 arguments: file_path, kA, kB, L");
        }
        //print arguments
        System.out.print("File path = " + args[0] + ", ");
        System.out.print("KA = " + args[1]+ ", ");
        System.out.print("KB = " + args[2]+ ", ");
        System.out.println("L = " + args[3]);

        //set variable to respective args
        int ka = Integer.parseInt(args[1]);
        int kb = Integer.parseInt(args[2]);
        int L = Integer.parseInt(args[3]);

        //validate ka, kb and L
        if ((ka == 0 && kb == 0) || L <= 0) {
            throw new IllegalArgumentException("Error: ka and kb cannot both be 0, and L must be positive");
        }

        //mute Spark log output
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);

        //initialize Spark
        SparkConf conf = new SparkConf(true).setAppName("G35HW1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("OFF");

        //read input file into RDD and split into L partitions
        JavaRDD<String> textFile = sc.textFile(args[0],L);

        //parse each line into a (Vector, group) pair
        JavaRDD<Tuple2<Vector, String>> inputPoints = textFile.map(line -> {
            //split line into coordinates and group label
            String[] parts = line.split(",");
            double[] coordinates = new double[parts.length-1];
            //put point's coord into coordinates from parts
            for(int i=0; i<parts.length-1; i++){
                coordinates[i] = Double.parseDouble(parts[i]);
            }
            //parse array into Vector
            Vector pointCoords = Vectors.dense(coordinates);
            //return coords and relative label
            return new Tuple2<>(pointCoords, parts[parts.length-1]);
        });

        //avoid recomputation on multiple actions
        inputPoints.cache();

        //count total points and points per group
        long n = inputPoints.count();
        long na = inputPoints.filter(point -> point._2().equals("A")).count();
        long nb = inputPoints.filter(point -> point._2().equals("B")).count();


        System.out.print("N = "+ n + ", ");
        System.out.print("NA = "+ na + ", ");
        System.out.println("NB = " + nb);

        //validate ka > na and kb > nb for FairFFT
        if (ka > na || kb > nb) {
            throw new IllegalArgumentException("Error: ka must be <= number of A points and kb must be <= number of B points");
        }

        //run MRFairFFT and measure execution time
        long time = System.currentTimeMillis();
        ArrayList<Tuple2<Vector, String>> s = MRFairFFT(inputPoints,ka,kb,L);
        time = System.currentTimeMillis() - time;

        //print centers with group label
        for(int i=0; i<s.size(); i++){
            System.out.println("Center = "+ s.get(i)._1() +" Label = "+s.get(i)._2());
        }

        //for each point in U, compute the distance to its nearest center in S
        JavaRDD<Double> minDistances = inputPoints.map(point -> {
            //start with distance to first center
            double minD = distance(point._1(), s.get(0)._1());
            //check all other centers and keep the minimum distance
            for(int i = 1; i < s.size(); i++){ // i=1 because minD is already calculated in first point
                double d = distance(point._1(), s.get(i)._1());
                if(d < minD){
                    minD = d;
                }
            }
            return Math.sqrt(minD); //sqrt because I work with square distance
        });

        //maximum over all points of their distance to nearest center
        double maxD = minDistances.reduce((a, b) -> Math.max(a, b));

        //print objective function value and execution time
        System.out.println("Objective function = "+maxD);
        System.out.println("Running time of MRFairFFT = "+time+" ms");
    }
}