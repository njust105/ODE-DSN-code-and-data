


### Environment variables
- `LIBTORCH_DIR`: path to libtorch directory

### Build and compile
```bash
git clone https://github.com/nlohmann/json.git
git clone https://github.com/d99kris/rapidcsv.git
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build/release
cmake --build build/release
```

### Useage:
- training
```bash
build/release/train-astress-opt configure.json
```

- training using Bayesian optimization
```bash
python scripts/optim_parameters.py
```

