"""
Lattice gauge theory utilities for the Davis-Wilson framework.

This package provides:
- SU(3) matrix operations
- Gauge configuration I/O and generation
- Wilson loop computation
- Topological charge measurement
- Skeleton construction for the Davis-Wilson map
"""

from .su3 import (
    random_su3,
    is_su3,
    su3_dagger,
    su3_multiply,
)
from .gauge_config import (
    GaugeConfig,
    load_config,
    save_config,
    generate_config_hmc,
    cold_start,
    hot_start,
)
from .wilson_loops import (
    compute_plaquette,
    compute_wilson_loop,
    compute_all_plaquettes,
    average_plaquette,
)
from .topological import (
    compute_field_strength_clover,
    compute_topological_charge,
    apply_smearing,
    apply_wilson_flow,
)
from .skeleton import (
    SkeletonSpec,
    build_skeleton,
    build_hierarchical_skeleton,
    build_block_skeleton,
    coarse_grain_config,
)
from .wilson_loops import Loop
from .backends import (
    read_openqcd_config,
    read_grid_config,
    read_milc_config,
    read_ildg_config,
    load_configs,
    convert_to_hdf5,
    run_openqcd,
    download_milc_configs,
)

__all__ = [
    # SU(3)
    "random_su3",
    "is_su3", 
    "su3_dagger",
    "su3_multiply",
    # Config
    "GaugeConfig",
    "load_config",
    "save_config",
    "generate_config_hmc",
    "cold_start",
    "hot_start",
    # Wilson
    "compute_plaquette",
    "compute_wilson_loop",
    "compute_all_plaquettes",
    "average_plaquette",
    # Topological
    "compute_field_strength_clover",
    "compute_topological_charge",
    "apply_smearing",
    "apply_wilson_flow",
    # Skeleton
    "Loop",
    "SkeletonSpec",
    "build_skeleton",
    "build_hierarchical_skeleton",
    "build_block_skeleton",
    "coarse_grain_config",
    # Backends
    "read_openqcd_config",
    "read_grid_config",
    "read_milc_config",
    "read_ildg_config",
    "load_configs",
    "convert_to_hdf5",
    "run_openqcd",
    "download_milc_configs",
]
