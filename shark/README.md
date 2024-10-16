

This is a MOOSE-based application.
- Follow [instructions](https://mooseframework.inl.gov/getting_started/installation/) to install MOOSE.
- Complie in the current directory:
    ```bash
    make -j4
    ```


- To generate random strain input sequences:
    ```bash
    pip install numpy scipy matplotlib pandas fastdtw
    python scripts/random_series.py
    ```
    Results will be in the `paths/` directory. 


- The examples are in the directory [./examples](./examples).
    Put the `paths` into `examples/<example-name>/`:
    ```bash
    mv paths examples/<example-name>/
    ```
    Batch computation of the RVE problem for all strain input sequences:
    ```bash
    cd examples/<example-name>
    ./run.sh
    ```
    The CSV output files will be in the `examples/<example-name>/Outputs` directory.