# Using Production Backends for Configuration Generation

The HMC algorithm in this repo is a placeholder. For real physics, use battle-tested codes.

## Option 1: Download Public Configs (Easiest)

### MILC Collaboration
```bash
# Download asqtad configs (2+1 flavor, but pure gauge sector is what we need)
python -c "
from lattice.backends import download_milc_configs
paths = download_milc_configs('asqtad_2064f21b676m010m050', './configs/milc', n_configs=100)
print(f'Downloaded {len(paths)} configs')
"
```

### ILDG (International Lattice Data Grid)
Browse: https://hpc.desy.de/ildg/

```bash
# Requires registration, then use ildg-tools
pip install ildg-tools
ildg-get --ensemble "qcdsf/..." --output ./configs/ildg/
```

### Lattice Data Archive
NERSC: https://portal.nersc.gov/project/m888/
Jülich: https://datapub.fz-juelich.de/qcd/

## Option 2: openQCD (Recommended for Pure Gauge)

openQCD is Martin Lüscher's production code. Gold standard for pure SU(3).

### Installation

```bash
# Clone
git clone https://github.com/luscher/openQCD.git
cd openQCD

# Configure for your system
# Edit Makefile: set MPI compiler, flags, etc.

# Compile
make -j8

# Test
cd main
mpirun -np 4 ./ym1 -i ../tests/ym1/sample.in
```

### Generate Configs

```bash
# Create input file for your parameters
cat > davis_run.in << 'EOF'
[Run name]
name         davis_L16_b6.0

[Directories]
log_dir      ./log
dat_dir      ./dat  
loc_dir      ./loc
cnfg_dir     ./cnfg

[Lattice parameters]
beta         6.0
c0           1.0

[Geometry]
L0           16
L1           16
L2           16
L3           16

[Boundary conditions]
type         3

[HMC parameters]
actions      0 1 2
integrator   OMF4
lambda       0.19
tau          1.0
nstep        20

[Level 0]
action       0 1 2
force        0 1 2

[Trajectory number]
nth          500
ntr          10000
nfr          10

[Random number generator]
seed         12345
EOF

# Run
mkdir -p log dat loc cnfg
mpirun -np 4 ./ym1 -i davis_run.in

# Configs saved to ./cnfg/
ls cnfg/
```

### Convert to Our Format

```python
from pathlib import Path
from lattice.backends import convert_to_hdf5

# Convert openQCD configs to HDF5
input_paths = sorted(Path("./cnfg").glob("*.cnfg"))
convert_to_hdf5(
    source="openqcd",
    input_paths=input_paths,
    output_dir="./configs/openqcd_converted",
    beta=6.0
)
```

## Option 3: Grid (More Flexible)

Grid is Peter Boyle's modern lattice QCD library. Better GPU support.

### Installation

```bash
# Dependencies
sudo apt install libgmp-dev libmpfr-dev libfftw3-dev libopenmpi-dev

# Clone and build
git clone https://github.com/paboyle/Grid.git
cd Grid
./bootstrap.sh
mkdir build && cd build
../configure --enable-precision=double --enable-simd=AVX2
make -j8
make install
```

### Generate Configs (C++)

```cpp
// hmc_wilson.cc
#include <Grid/Grid.h>

using namespace Grid;

int main(int argc, char **argv) {
    Grid_init(&argc, &argv);
    
    const int L = 16;
    Coordinate latt({L, L, L, L});
    Coordinate simd({1, 1, 1, 1});
    Coordinate mpi({1, 1, 1, 1});
    
    GridCartesian Grid(latt, simd, mpi);
    GridRedBlackCartesian RBGrid(&Grid);
    
    // RNG
    GridParallelRNG pRNG(&Grid);
    pRNG.SeedFixedIntegers({1, 2, 3, 4});
    
    // Gauge field
    LatticeGaugeField U(&Grid);
    SU<Nc>::HotConfiguration(pRNG, U);
    
    // Wilson action
    RealD beta = 6.0;
    WilsonGaugeActionR action(beta);
    
    // Integrator
    IntegratorParameters params(1, 1.0);  // 1 trajectory, tau=1
    MinimumNorm2<GaugeStatistics> integrator(action, params);
    
    // HMC
    HMCparameters HMCparams;
    HMCparams.StartTrajectory = 0;
    HMCparams.Trajectories = 10000;
    HMCparams.NoMetropolisUntil = 500;  // Thermalization
    
    HybridMonteCarlo<MinimumNorm2<GaugeStatistics>> HMC(HMCparams, integrator, pRNG);
    
    // Checkpointer
    CheckpointerParameters CPparams;
    CPparams.config_prefix = "config";
    CPparams.saveInterval = 10;
    BinaryHmcCheckpointer<WilsonGaugeActionR> CP(CPparams);
    
    HMC.AddObservable(&CP);
    HMC.evolve(U);
    
    Grid_finalize();
    return 0;
}
```

```bash
# Compile and run
g++ -O3 hmc_wilson.cc -o hmc_wilson $(Grid-config --cxxflags --ldflags --libs)
mpirun -np 4 ./hmc_wilson
```

### Convert to Our Format

```python
from pathlib import Path
from lattice.backends import convert_to_hdf5

input_paths = sorted(Path(".").glob("config.*"))
convert_to_hdf5(
    source="grid",
    input_paths=input_paths,
    output_dir="./configs/grid_converted",
    L=16,
    beta=6.0
)
```

## Option 4: Modal + Pre-generated Configs

If you have configs on disk or S3, mount them in Modal:

```python
# modal_app.py addition

# Mount local configs
local_configs = modal.Mount.from_local_dir(
    "./configs",
    remote_path="/data/configs"
)

@app.function(mounts=[local_configs])
def analyze_external_configs():
    from pathlib import Path
    from lattice.backends import load_configs
    
    paths = sorted(Path("/data/configs").glob("*.cnfg"))
    configs = load_configs("openqcd", paths, beta=6.0)
    
    # Run analysis...
```

Or from S3:

```python
# Mount S3 bucket with configs
s3_configs = modal.CloudBucketMount(
    bucket_name="my-lattice-configs",
    secret=modal.Secret.from_name("aws-secret"),
)

@app.function(cloud_bucket_mounts={"/data/s3": s3_configs})
def analyze_s3_configs():
    # Same as above
    pass
```

## Recommended Workflow

1. **Development**: Use small lattices (8³×8, 12³×12) generated with openQCD locally
2. **Production**: Use 16³×16 or 24³×24 configs from MILC/ILDG public datasets
3. **Publication**: Generate fresh configs with openQCD/Grid at target parameters

## Beta Values for SU(3)

| β | a (fm) | Phase | Use Case |
|---|--------|-------|----------|
| 5.7 | ~0.17 | Confined | Strong coupling tests |
| 6.0 | ~0.10 | Confined | Standard production |
| 6.2 | ~0.07 | Confined | Finer lattice |
| 6.5 | ~0.05 | Near transition | Scaling tests |
| 7.0+ | ~0.03 | Weak coupling | Perturbative regime |

For the deconfinement transition on finite lattices:
- N_t = 4: β_c ≈ 5.69
- N_t = 6: β_c ≈ 5.89
- N_t = 8: β_c ≈ 6.06

## File Formats Quick Reference

| Format | Extension | Reader | Notes |
|--------|-----------|--------|-------|
| openQCD | .cnfg | `read_openqcd_config` | Martin Lüscher's format |
| Grid | .bin | `read_grid_config` | Needs L parameter |
| MILC | varies | `read_milc_config` | Big/little endian |
| ILDG | .lime | `read_ildg_config` | Universal standard |
| Ours | .h5 | `load_config` | HDF5, compressed |

## Troubleshooting

**"Configs don't load"**: Check endianness. MILC varies by platform.

**"Memory error"**: 24⁴ configs are ~760MB each. Use streaming.

**"Wrong plaquette"**: Verify β matches what was used in generation.

**"Topological charge not integer"**: Apply Wilson flow before measuring.
