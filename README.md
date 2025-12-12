This repository is a temporary workspace only for demonstrating the workload completed for CSCI 775.

The general usage is

```bash
./main <prefix> <output_file> <0/1> [optional: log_shots]
```

<0/1> for whether to output the result in shell.

For example, to run a simulation of ghz_n40 with $2^{20}$ shots, execute:

```bash
./main benchmarks/qasm/ghz ghz_n40.qasm results/result.csv 1 20
```

To run a batch of simulation of ghz, execute:

```bash
./main benchmarks/qasm/ghz ghz results/result.csv 1 20
```
