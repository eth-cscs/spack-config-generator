#!/usr/bin/env spack-python

# Note:
# Due to a limitation of spack-python, this is a single script.
# spack-python gives access to spack python library, but it makes
# difficult to import custom libraries.
# This is mainly due to how it works, since it emulates a python
# interpreter via InteractiveInterpreter.

import re
import argparse
from pathlib import Path

from spack.version import Version
from spack.spec import Spec, CompilerSpec
from spack.detection import DetectedPackage
import spack.util.spack_yaml as syaml
import spack.repo
import spack.detection


class Module:
    def __init__(self, namever: str):
        self.fullname = namever
        self.name, self.version = self.fullname.split("/")

    def __repr__(self):
        return self.fullname


def parse_modulerc(modulerc_path: Path):
    modules = []

    with open(modulerc_path, "r") as modulerc_file:
        for module in modulerc_file.readlines():
            if not module.startswith(r"module-version"):
                continue
            module_namever = module.split()[1]
            modules.append(Module(module_namever))

    return modules


def parse_lua_modulerc(modulerc_path: Path):
    modules = []

    with open(modulerc_path, "r") as modulerc_file:
        for module in modulerc_file.readlines():
            if not module.startswith(r"module_version"):
                continue
            module_namever = module[15:].split(',')[0].replace("\"", "")
            modules.append(Module(module_namever))

    return modules


VALID_COMPILERS = [
    "aocc",
    "cce",
    "clang",
    "gcc",  # TODO gcc-cross-aarch
    "intel",  # TODO intel-classic, intel-oneapi, ... (on eiger)
    "pgi",
    "nvidia",
    "rocm-compiler",
]


CRAY_PACKAGES = [
    "cray-jemalloc",
    "cray-R",
    # "cray-fftw", # dropped, because of cray compiler wrapper issues depending on whether cray-mpich is loaded first or not
    "cray-mpich",  # TODO cray-mpich-abi, cray-mpich-ucx, ... ??
    "cray-python",
    "cray-netcdf",
    # "cray-hdf5", # dropped, because spack does not detect the prefix path correctly.
    "cray-petsc",
    "cray-libsci",
    "papi",
]


CRAY2SPACK = {
    "cray-R": ("r", ""),
    "cray-hdf5": ("hdf5", "~mpi+hl+fortran"),
    "cray-hdf5-parallel": ("hdf5", "+mpi+hl+fortran"),
    "cray-jemalloc": ("jemalloc", ""),
    "cray-mpich": ("cray-mpich", ""),
    "cray-libsci": ("cray-libsci", ""),
    "cray-fftw": ("cray-fftw", ""),
    "cray-netcdf-c": ("netcdf-c", "~parallel-netcdf+mpi"),
    "cray-netcdf-fortran": ("netcdf-fortran", ""),
    "cray-netcdf-hdf5parallel": ("netcdf-c", "+parallel-netcdf+mpi"),
    "cray-netcdf-hdf5parallel-c": ("netcdf-c", "+parallel-netcdf+mpi"),
    "cray-netcdf-hdf5parallel-fortran": ("netcdf-fortran", ""),
    "cray-netcdf-parallel": ("netcdf-c", "+cxx+fortran"),
    "cray-petsc": ("petsc", "~int64~complex~cuda"),
    "cray-petsc-64": ("petsc", "+int64~complex~cuda"),
    "cray-petsc-complex": ("petsc", "~int64+complex~cuda"),
    "cray-petsc-complex-64": ("petsc", "+int64+complex~cuda"),
    "papi": ("papi", ""),
}


# "Systems with newer CrayPE (21.10 for EX systems, future work for CS and
#  XC systems) have compilers and MPI wrappers that can be used directly
#  by path. These systems are considered ``linux`` platforms."
#
# Hohgant is detected as linux, while Daint is detected as cray. This has
# effects on how compilers are detected.
#
# See:
# https://github.com/spack/spack/blob/ca265ea0c268d1be05c85e66b25916e0d8c85932/lib/spack/spack/platforms/cray.py#L142-L162
import spack.platforms
host = str(spack.platforms.host())


class CrayPE:
    """
    CrayPE (CPE) / CrayDT (CDT)
    """

    def __init__(self, name, version, modules: [(str, str)]):
        self._name = name
        self._version = version
        self._setup_modules(modules)

    def __repr__(self):
        return f"{self._name}-{self._version}"

    def _setup_modules(self, modules):
        """
        self._packages:     modules for packages
        self._compilers:    modules for compilers
        """

        def is_compiler(module: Module):
            return module.name in VALID_COMPILERS

        def is_package(module: Module):
            return any([module.name.startswith(prefix) for prefix in CRAY_PACKAGES])

        def is_interesting(module: Module):
            is_prgenv = module.name.startswith("PrgEnv-")
            return not is_prgenv and (is_package(module) or is_compiler(module))

        all_modules = [module for module in modules if is_interesting(module)]
        self._packages = [module for module in all_modules if is_package(module)]
        self._compilers = [module for module in all_modules if is_compiler(module)]

    def _generate_packages(self):
        packages = []
        for module in self._packages:
            if not module.name in CRAY2SPACK.keys():
                print("?", "skipping", module)
                continue

            spec_txt = " ".join(CRAY2SPACK[module.name])

            # Note:
            # `cray` is a required module to enable any other module
            required_modules = ["cray", module.fullname]

            # Note:
            # This is a workaround, since `cray-mpich` has `libfabric` as dependency,
            # but it does not load it.
            if module.name == "cray-mpich":
                required_modules.append("libfabric")

            spec = Spec(
                f"{spec_txt}@{module.version}", external_modules=required_modules
            )
            packages.append(DetectedPackage(spec, None))
        return packages

    def _generate_compilers(self, all_compilers):
        from spack.compilers import _to_dict

        compilers = []
        for compiler_module in self._compilers:
            def match(c):
                # nVidia HPC module is called 'nvidia', while Spack calls it 'nvhpc'
                compiler_module_name = compiler_module.name.replace('nvidia', 'nvhpc')
                return (
                        c.name == compiler_module_name and
                        c.version >= Version(compiler_module.version)
                        )

            if host == "cray":
                available_compilers = all_compilers[:]

                found_compilers = [c for c in available_compilers if match(c) and len(c.modules) > 0]

                # TODO:
                # currently spack detects nvhpc compilers based on the
                # vanilla nvhpc modules generated by NVIDIA.
                # But Cray of course has its own terribleness and only
                # provides PrgEnv-nvidia, which uses nvidia/xyz, which is
                # a modified module which also sets CRAY_* env variables
                # to commit crimes in their compiler wrapper.
                # We really want to only provide what's in the CDT, i.e.
                # PrgEnv-nvidia with nvidia modules.
                if compiler_module.name == 'nvidia':
                    for fc in found_compilers:
                        fc.modules = [m.replace("nvhpc", "nvidia") for m in fc.modules]
            elif host == "linux":
                from spack.util.module_cmd import path_from_modules

                available_compilers = []
                try:
                    path = path_from_modules([str(compiler_module)])
                    if path:
                        available_compilers = find_compilers([path])
                except Exception as e:
                    print(e)

                found_compilers = [c for c in available_compilers if match(c)]

            if len(found_compilers) == 0:
                print("?", f"compiler {compiler_module} not found")
                continue

            for found_compiler in found_compilers:
                compilers.append(found_compiler)

        compilers = sorted(compilers, key=lambda c: (c.name, c.version))

        return [_to_dict(c) for c in compilers]


def all_craypes():
    all_cpes = []
    for modulerc_path in Path(r"/opt/cray").rglob(r"modulerc"):
        name, version = reversed([parent.name for parent in modulerc_path.parents][:2])
        modules = parse_modulerc(modulerc_path)
        all_cpes.append(CrayPE(name, version, modules))
    return all_cpes


def detect_mkl():
    libs_search_paths = {
        "intel-mkl": Path("/opt/intel").glob(r"compilers_and_libraries_*/linux/mkl"),
        "intel-oneapi-mkl": Path("/opt/intel/oneapi/mkl").glob(r"*"),
    }

    mkl_pkgs = []
    for libname, paths in libs_search_paths.items():
        for mkl_root in paths:
            if mkl_root.is_symlink():
                continue

            version = re.search(r"(\d+\.\d+\.\d+)", str(mkl_root)).group(1)
            mkl_pkgs.append(
                DetectedPackage(Spec(f"{libname}@{version}"), mkl_root.as_posix())
            )
    return mkl_pkgs


def detect_cuda():
    cuda_pkgs = []
    for default_install_path in ["/usr/local", "/opt/nvidia"]:
        # note: cuda*/** is needed because of a strange installation path
        # which I don't know if it is an old standard one
        for nvcc_exec in Path(default_install_path).rglob("cuda*/**/bin/nvcc"):
            cuda_root = nvcc_exec.parents[1]  # where is bin?

            # it's a symlink, so it will be considered (hopefully) via its real position
            if cuda_root.is_symlink():
                continue

            # Note:
            # nVidia HPC SDK has a different folder structure, and it may contain different
            # CUDA toolkits. FindCUDA from CMake seems not aware of it (e.g. cuBLAS in math_libs).
            if "hpc_sdk" in cuda_root.parts:
                continue

            # parse version from $CUDA_HOME/version.txt
            with open(cuda_root / "version.txt", "r") as version_file:
                version = re.match(
                    r"CUDA Version (\d+\.\d+\.\d+)", version_file.read()
                ).group(1)

            cuda_pkgs.append(
                DetectedPackage(Spec(f"cuda@{version}"), cuda_root.as_posix())
            )
    return cuda_pkgs

def detect_executables():
    # Cray's compiler wrappers depend on pkg-config, and in particular
    # some packages Cray puts on the system have their pc files in a default search
    # path, meaning that whenever Spack builds pkg-config, and you invoke the
    # Cray compiler wrapper, or even module unload xyz, it fails, since Spack's
    # pkg-config has different default paths. So, the easiest solution is to
    # define pkg-config as an external and hope Spack uses this one by default.
    return spack.detection.by_executable([spack.repo.path.get_pkg_class('pkg-config')])['pkg-config']


def to_config_data(packages):
    # [DetectedPacakge] -> {package_name: [DetectedPackage]}
    from collections import defaultdict

    detected_packages = defaultdict(list)
    for pkg in packages:
        try:
            Spec.ensure_valid_variants(pkg.spec)
        except Exception as e:
            print("!", e)
            continue

        detected_packages[pkg.spec.name].append(pkg)

    # Generate well-formed config data for pacakges.yaml
    # (source: spack.detection.update_configuration)
    from spack.detection.common import _pkg_config_dict

    buildable = True
    pkg_to_cfg = {
        "all": {
            "providers": {
                "mpi": ["cray-mpich"],
                "pkgconfig": ["pkg-config", "pkgconf"]
            }
        }
    }
    for package_name, entries in detected_packages.items():
        pkg_config = _pkg_config_dict(entries)
        # Only keep a prefix
        for external in pkg_config["externals"]:
            if "modules" in external and "prefix" in external:
                del external["prefix"]
        if buildable is False:
            pkg_config["buildable"] = False
        pkg_to_cfg[package_name] = pkg_config

    return {"packages": pkg_to_cfg}


if __name__ == "__main__":
    from spack.compilers import find_compilers

    parser = argparse.ArgumentParser(description='Generate spack config.')
    parser.add_argument('--current-cpe',
        dest='just_current_cpe',
        action='store_true',
        default=False,
        help='Whether to look for the CPE currently loaded, or for all the ones available on the system.')
    parser.add_argument('--output-folder,-o',
        dest='output_path',
        default="./generated-configs",
        type=Path,
        help='Where to store the configuration files generated.')
    args = parser.parse_args()

    available_compilers = find_compilers()

    pkgs = []
    #pkgs.extend(detect_mkl())
    pkgs.extend(detect_cuda())
    pkgs.extend(detect_executables())

    def generate_cpe_config(cpe, cpe_configs_path):
        print("\t\t", cpe)

        cpe_pkgs = pkgs.copy()
        cpe_pkgs.extend(cpe._generate_packages())

        cpe_configs_path.mkdir(parents=True, exist_ok=True)

        with open(cpe_configs_path / "packages.yaml", "w") as yaml_file:
            syaml.dump_config(to_config_data(cpe_pkgs), yaml_file)

        cpe_compilers = cpe._generate_compilers(available_compilers)
        with open(cpe_configs_path / "compilers.yaml", "w") as yaml_file:
            syaml.dump_config({"compilers": cpe_compilers}, yaml_file)

    if args.just_current_cpe:
        import os
        import os.path
        from pathlib import PosixPath

        ENV_VARIABLE="LMOD_MODULERCFILE"
        if ENV_VARIABLE not in os.environ.keys():
            raise ValueError(f"No {ENV_VARIABLE} found. Check if a CPE is loaded with `module list`.")
        lmods = os.environ["LMOD_MODULERCFILE"]

        modulerc_path = PosixPath(lmods.split(':')[0])
        PE_ROOT = PosixPath('/opt/cray/pe')

        cpe_name, cpe_version = modulerc_path.relative_to(PE_ROOT).parts[:-1]

        cpe = CrayPE(cpe_name, cpe_version, parse_lua_modulerc(modulerc_path))
        generate_cpe_config(cpe, Path(f"{args.output_path.expanduser()}"))
    else:
        for cpe in all_craypes():
            generate_cpe_config(cpe, Path(f"{args.output_path.expanduser()}/{cpe}"))
