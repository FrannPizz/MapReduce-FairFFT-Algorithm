# Fair *k*-Center Clustering with MapReduce (Spark)

Big Data Computing, University of Padua.

A 2-round, coreset-based MapReduce strategy for **fair *k*-center clustering**, implemented in Java on Apache Spark and tested on the CloudVeneto cluster.

## Problem

Given a *2-labeled pointset* `U ⊆ ℝ^D` — where every point carries a demographic label `g ∈ {A, B}` — and two budgets `kA`, `kB`, select a set `S ⊆ U` of `k = kA + kB` centers containing **exactly** `kA` points from group `A` and `kB` from group `B`, so as to minimize the covering radius:

```
min  max_{x ∈ U}  dist(x, S)
```

where `dist` is the Euclidean distance. This is the standard *k*-center objective with an added **fairness constraint** on the composition of the center set.

## Algorithms

### `FairFFT` — sequential
A variant of Farthest-First Traversal that tracks the demographic group of each selected center. Once the budget for a group is exhausted (`kA` centers from `A` or `kB` from `B`), the remaining iterations pick centers only from the other group, guaranteeing the exact `(kA, kB)` split.

### `MRFairFFT` — MapReduce (2 rounds)
- **Round 1 (map):** each Spark partition runs `FairFFT` locally to extract a coreset. To improve accuracy, each partition is allowed to return up to `kA·L` + `kB·L` centers rather than just `kA + kB`.
- **Round 2 (reduce):** the union of all partition coresets is collected to the driver and `FairFFT` is run once more to produce the final `k` centers.

The input set `U` is kept exclusively in distributed form (RDD); only the small coreset is ever collected to the driver.

## Project structure

```
src/main/java/G35HW1.java     # Fair k-center: FairFFT + MRFairFFT + driver
build.gradle                  # Gradle build, produces a fat JAR
```

## Input format

A text file with one point per line: the `D` real coordinates separated by commas, followed by the group label `A` or `B`.

```
1.5,6.0,2.3,A
0.7,4.1,9.2,B
```

## Build

The build produces a self-contained JAR (`DBC.jar`) with Spark dependencies marked `compileOnly` (provided by the cluster at runtime).

```bash
./gradlew jar          # Linux / macOS
gradlew.bat jar        # Windows
```

The JAR is written to `build/libs/DBC.jar`.

## Run

```
spark-submit --class G35HW1 build/libs/DBC.jar <file_path> <kA> <kB> <L>
```

| Argument    | Meaning                                              |
|-------------|------------------------------------------------------|
| `file_path` | path to the input file (local or HDFS)               |
| `kA`        | number of centers to pick from group A               |
| `kB`        | number of centers to pick from group B               |
| `L`         | number of partitions of the input RDD                |

**Local example:**

```bash
spark-submit --master "local[*]" --class G35HW1 build/libs/DBC.jar input.txt 3 2 4
```

**Cluster (CloudVeneto, YARN):** the master is set automatically — do **not** force `local[*]`.

```bash
spark-submit --num-executors 16 --class G35HW1 BDC.jar /data/BDC2526/dataset.txt 3 2 8
```

## Output

```
File path = input.txt, KA = 3, KB = 2, L = 4
N = 10000, NA = 6000, NB = 4000
Center = [...] Label = A
...
Objective function = 12.345678
Running time of MRFairFFT = 87 ms
```

The reported running time covers only `MRFairFFT`, excluding dataset loading.

## Implementation notes

- Points are represented as `org.apache.spark.mllib.linalg.Vector` (the `mllib` package, **not** `ml`).
- Distances are computed with `Vectors.sqdist` (squared Euclidean) throughout the inner loops; the square root is taken only once when reporting the final objective value.
- `inputPoints` is cached, since it is reused by `count`, the per-group filters, and the final objective computation.

## Notes

Developed and tested in local mode, then run on the CloudVeneto cluster (10 machines, 8 cores / 16 GB RAM each) with datasets preloaded in HDFS under `/data/BDC2526`.
