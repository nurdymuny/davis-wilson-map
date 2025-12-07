"""
Production backends for gauge configuration generation.

Instead of implementing HMC from scratch, we interface with battle-tested codes:
- openQCD: https://luscher.web.cern.ch/luscher/openQCD/
- Grid: https://github.com/paboyle/Grid

These are what actual lattice QCD collaborations use.
"""

from __future__ import annotations

import subprocess
import tempfile
import struct
from pathlib import Path
from typing import Optional, Literal
import numpy as np

from .gauge_config import GaugeConfig, save_config


# =============================================================================
# OpenQCD Interface
# =============================================================================

OPENQCD_TEMPLATE = """
[Run name]
name         {run_name}

[Directories]
log_dir      {output_dir}/log
dat_dir      {output_dir}/dat
loc_dir      {output_dir}/loc
cnfg_dir     {output_dir}/cnfg

[Lattice parameters]
beta         {beta}
c0           1.0
kappa        0.0
csw          0.0

[Geometry]
L0           {L}
L1           {L}
L2           {L}
L3           {L}

[Boundary conditions]
type         3

[HMC parameters]
actions      0 1 2
integrator   {integrator}
lambda       0.19
tau          {trajectory_length}
nstep        {n_steps}

[Level 0]
action       0 1 2
force        0 1 2

[Trajectory number]
nth          {thermalization}
ntr          {n_trajectories}
nfr          {save_frequency}

[Random number generator]
seed         {seed}
"""


def generate_openqcd_input(
    L: int,
    beta: float,
    n_trajectories: int,
    output_dir: Path,
    thermalization: int = 100,
    trajectory_length: float = 1.0,
    n_steps: int = 20,
    seed: int = 12345,
) -> str:
    """Generate openQCD input file."""
    return OPENQCD_TEMPLATE.format(
        run_name=f"davis_L{L}_b{beta}",
        output_dir=str(output_dir),
        beta=beta,
        L=L,
        integrator="OMF4",  # 4th order Omelyan integrator
        trajectory_length=trajectory_length,
        n_steps=n_steps,
        thermalization=thermalization,
        n_trajectories=n_trajectories,
        save_frequency=1,
        seed=seed,
    )


def read_openqcd_config(path: Path) -> GaugeConfig:
    """
    Read openQCD configuration file (.cnfg format).
    
    openQCD stores configs in a custom binary format:
    - Header: lattice dimensions, plaquette, etc.
    - Data: SU(3) matrices in row-major order, double precision
    
    Reference: openQCD/modules/archive/archive.c
    """
    with open(path, "rb") as f:
        # Read header
        header = f.read(64)
        L0, L1, L2, L3 = struct.unpack("4i", header[:16])
        
        if not (L0 == L1 == L2 == L3):
            raise ValueError(f"Non-cubic lattice not supported: {L0}x{L1}x{L2}x{L3}")
        
        L = L0
        
        # Read gauge field
        # openQCD stores: U[t][x][y][z][mu] as 3x3 complex matrices
        n_links = 4 * L**4
        n_elements = n_links * 18  # 9 complex = 18 real per SU(3)
        
        data = np.frombuffer(f.read(n_elements * 8), dtype=np.float64)
        data = data.reshape(L, L, L, L, 4, 3, 3, 2)  # last dim: real/imag
        
        # Convert to complex and reorder to our convention
        U_complex = data[..., 0] + 1j * data[..., 1]
        
        # Reorder from [t,x,y,z,mu,...] to [mu,t,x,y,z,...]
        U = np.transpose(U_complex, (4, 0, 1, 2, 3, 5, 6))
    
    return GaugeConfig(
        U=U.astype(np.complex128),
        L=L,
        beta=0.0,  # Will be set from metadata
        metadata={"source": "openQCD", "file": str(path)}
    )


def run_openqcd(
    L: int,
    beta: float,
    n_configs: int,
    output_dir: Path,
    openqcd_path: Path = Path("/opt/openQCD/main"),
    mpi_ranks: int = 4,
    thermalization: int = 100,
) -> list[Path]:
    """
    Run openQCD to generate configurations.
    
    Requires openQCD to be installed and compiled.
    
    Args:
        L: Lattice size
        beta: Gauge coupling
        n_configs: Number of configurations to generate
        output_dir: Where to save configs
        openqcd_path: Path to openQCD main directory
        mpi_ranks: Number of MPI processes
        thermalization: Trajectories to discard
    
    Returns:
        List of paths to generated configuration files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    for subdir in ["log", "dat", "loc", "cnfg"]:
        (output_dir / subdir).mkdir(exist_ok=True)
    
    # Write input file
    input_content = generate_openqcd_input(
        L=L,
        beta=beta,
        n_trajectories=n_configs,
        output_dir=output_dir,
        thermalization=thermalization,
    )
    
    input_file = output_dir / "input.in"
    with open(input_file, "w") as f:
        f.write(input_content)
    
    # Run openQCD
    exe = openqcd_path / "ym1"
    cmd = [
        "mpirun", "-np", str(mpi_ranks),
        str(exe), "-i", str(input_file)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"openQCD failed:\n{result.stderr}")
    
    # Find generated configs
    cnfg_dir = output_dir / "cnfg"
    config_paths = sorted(cnfg_dir.glob("*.cnfg"))
    
    print(f"Generated {len(config_paths)} configurations")
    return config_paths


# =============================================================================
# Grid Interface
# =============================================================================

GRID_SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""Grid configuration generation script."""

import Grid
from Grid import *

# Lattice setup
L = {L}
grid = Grid.GridCartesian([L, L, L, L], [1, 1, 1, 1])
rbgrid = Grid.GridRedBlackCartesian(grid)

# RNG
rng = Grid.GridParallelRNG(grid)
rng.SeedFixedIntegers([1, 2, 3, 4])

# Gauge field
U = Grid.LatticeGaugeField(grid)

# Start hot or cold
Grid.SU3.HotConfiguration(rng, U)

# Wilson action
beta = {beta}
action = Grid.WilsonGaugeActionR(beta)

# HMC parameters  
tau = {trajectory_length}
n_steps = {n_steps}

integrator = Grid.MinimumNorm2(action, tau / n_steps)
hmc = Grid.HybridMonteCarlo(
    grid, rng, action, integrator, tau
)

# Thermalization
print("Thermalizing...")
for i in range({thermalization}):
    hmc.evolve(U)
    if i % 10 == 0:
        plaq = Grid.WilsonLoops.avgPlaquette(U)
        print(f"Therm {{i}}: plaq = {{plaq}}")

# Production
print("Generating configurations...")
for i in range({n_configs}):
    hmc.evolve(U)
    
    # Save configuration
    writer = Grid.BinaryIO()
    writer.writeConfiguration(U, "{output_dir}/config_{{:05d}}.bin".format(i))
    
    plaq = Grid.WilsonLoops.avgPlaquette(U)
    print(f"Config {{i}}: plaq = {{plaq}}")

print("Done!")
'''


def generate_grid_script(
    L: int,
    beta: float,
    n_configs: int,
    output_dir: Path,
    thermalization: int = 100,
    trajectory_length: float = 1.0,
    n_steps: int = 20,
) -> str:
    """Generate Grid Python script for config generation."""
    return GRID_SCRIPT_TEMPLATE.format(
        L=L,
        beta=beta,
        n_configs=n_configs,
        output_dir=str(output_dir),
        thermalization=thermalization,
        trajectory_length=trajectory_length,
        n_steps=n_steps,
    )


def read_grid_config(path: Path, L: int) -> GaugeConfig:
    """
    Read Grid binary configuration file.
    
    Grid uses ILDG-like format or its own binary format.
    This reads the simple binary format from BinaryIO.
    """
    with open(path, "rb") as f:
        # Grid binary: just raw SU(3) matrices
        # Format: U[mu][site] where site is lexicographic
        n_links = 4 * L**4
        n_doubles = n_links * 18  # 9 complex per matrix
        
        data = np.frombuffer(f.read(n_doubles * 8), dtype=np.float64)
        data = data.reshape(4, L**4, 3, 3, 2)
        
        # Convert to complex
        U_flat = data[..., 0] + 1j * data[..., 1]
        
        # Reshape to [mu, t, x, y, z, 3, 3]
        U = U_flat.reshape(4, L, L, L, L, 3, 3)
    
    return GaugeConfig(
        U=U.astype(np.complex128),
        L=L,
        beta=0.0,
        metadata={"source": "Grid", "file": str(path)}
    )


# =============================================================================
# ILDG/LIME Format (Universal)
# =============================================================================

def read_ildg_config(path: Path) -> GaugeConfig:
    """
    Read ILDG (International Lattice Data Grid) configuration.
    
    ILDG uses LIME (Lattice Interoperability Message Encapsulation) format.
    This is the standard interchange format for lattice QCD configs.
    
    Reference: https://hpc.desy.de/ildg/
    
    For full ILDG support, use the `lime` library:
        pip install lime-lattice
    """
    try:
        import lime
    except ImportError:
        raise ImportError(
            "ILDG support requires the lime library:\n"
            "  pip install lime-lattice"
        )
    
    # Read LIME file
    reader = lime.Reader(str(path))
    
    # Find the gauge field record
    for record in reader:
        if record.type == "ildg-binary-data":
            # Parse header for dimensions
            # ILDG stores as: U[t][z][y][x][mu] in SU(3) row-major
            
            header = reader.get_record("ildg-format")
            # Parse XML header for dimensions...
            # (simplified - real implementation needs XML parsing)
            
            data = np.frombuffer(record.data, dtype=">f8")  # Big-endian doubles
            
            # Reshape and convert...
            # (implementation depends on specific ILDG variant)
            pass
    
    raise NotImplementedError("Full ILDG parsing requires more implementation")


# =============================================================================
# MILC Format
# =============================================================================

def read_milc_config(path: Path) -> GaugeConfig:
    """
    Read MILC collaboration configuration format.
    
    MILC uses a custom binary format with header.
    Reference: https://github.com/milc-qcd/milc_qcd
    """
    with open(path, "rb") as f:
        # MILC header (simplified)
        magic = struct.unpack(">I", f.read(4))[0]
        
        if magic == 0x4e66614c:  # "LfaN" (little-endian)
            endian = "<"
        elif magic == 0x4c61664e:  # "NafL" (big-endian)
            endian = ">"
        else:
            raise ValueError(f"Unknown MILC magic: {hex(magic)}")
        
        # Read dimensions
        dims = struct.unpack(f"{endian}4i", f.read(16))
        nx, ny, nz, nt = dims
        
        if not (nx == ny == nz == nt):
            raise ValueError(f"Non-cubic not supported: {dims}")
        
        L = nx
        
        # Skip rest of header
        f.seek(96)  # MILC header is 96 bytes
        
        # Read gauge field
        # MILC: U[t][z][y][x][mu] as SU(3), single precision
        n_links = 4 * L**4
        data = np.frombuffer(f.read(n_links * 18 * 4), dtype=f"{endian}f4")
        data = data.reshape(nt, nz, ny, nx, 4, 3, 3, 2)
        
        # Convert to complex128 and reorder
        U_complex = (data[..., 0] + 1j * data[..., 1]).astype(np.complex128)
        
        # Reorder to [mu, t, x, y, z, 3, 3]
        U = np.transpose(U_complex, (4, 0, 3, 2, 1, 5, 6))
    
    return GaugeConfig(U=U, L=L, beta=0.0, metadata={"source": "MILC", "file": str(path)})


# =============================================================================
# Public Dataset Download
# =============================================================================

MILC_DATASETS = {
    "asqtad_2064f21b676m010m050": {
        "url": "https://www.physics.utah.edu/~detar/milc/asqtad/2064f21b676m010m050/",
        "L": 20,
        "Lt": 64,
        "beta": 6.76,
        "description": "MILC asqtad 20^3 x 64, 2+1 flavor",
    },
    # Add more datasets as needed
}


def download_milc_configs(
    dataset: str,
    output_dir: Path,
    n_configs: int = 100,
) -> list[Path]:
    """
    Download public MILC configurations.
    
    Args:
        dataset: Name of MILC dataset (see MILC_DATASETS)
        output_dir: Where to save
        n_configs: How many to download
    
    Returns:
        List of paths to downloaded configs
    """
    import urllib.request
    
    if dataset not in MILC_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(MILC_DATASETS.keys())}")
    
    info = MILC_DATASETS[dataset]
    base_url = info["url"]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for i in range(n_configs):
        # MILC naming convention (varies by dataset)
        filename = f"l{info['L']}{info['Lt']}f21b{int(info['beta']*100)}.{i:04d}"
        url = f"{base_url}/{filename}"
        local_path = output_dir / filename
        
        if not local_path.exists():
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, local_path)
                paths.append(local_path)
            except Exception as e:
                print(f"  Failed: {e}")
                break
        else:
            paths.append(local_path)
    
    return paths


# =============================================================================
# Unified Interface
# =============================================================================

def load_configs(
    source: Literal["openqcd", "grid", "milc", "ildg"],
    paths: list[Path],
    L: Optional[int] = None,
    beta: Optional[float] = None,
) -> list[GaugeConfig]:
    """
    Load configurations from any supported format.
    
    Args:
        source: Which format ("openqcd", "grid", "milc", "ildg")
        paths: List of config file paths
        L: Lattice size (required for Grid format)
        beta: Coupling (added to metadata)
    
    Returns:
        List of GaugeConfig objects
    """
    readers = {
        "openqcd": read_openqcd_config,
        "grid": lambda p: read_grid_config(p, L) if L else read_grid_config(p, 16),
        "milc": read_milc_config,
        "ildg": read_ildg_config,
    }
    
    reader = readers[source]
    configs = []
    
    for path in paths:
        config = reader(Path(path))
        if beta is not None:
            config.beta = beta
        configs.append(config)
    
    return configs


def convert_to_hdf5(
    source: Literal["openqcd", "grid", "milc", "ildg"],
    input_paths: list[Path],
    output_dir: Path,
    L: Optional[int] = None,
    beta: Optional[float] = None,
) -> list[Path]:
    """
    Convert configs from external format to our HDF5 format.
    
    This is the bridge from production codes to our analysis pipeline.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configs = load_configs(source, input_paths, L, beta)
    output_paths = []
    
    for i, config in enumerate(configs):
        config.metadata["id"] = f"config_{i:05d}"
        config.metadata["original_file"] = str(input_paths[i])
        
        out_path = output_dir / f"config_{i:05d}.h5"
        save_config(config, out_path)
        output_paths.append(out_path)
    
    return output_paths
