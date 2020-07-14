# Vectorized Eigensolver

The vectorized-eigensolver allows for efficient computation of eigenvalues and eigenvectors
on modern hardware like multicore processors and general purpose graphics processing units.
To acieve good performance the data structure is changed and like the name suggests a
vectorization approach is used.

## Usage
Since this project is not very large and intended to be used as a header library, no real
installation is required. Just download the files and compile.
There are different compilation options

### OpenMP using GCC
In this version the GNU compiler and OpenMP for multiprocessing are used. Maybe you want to
adjust some options for best performance on your CPU.

For compilation type
```
make gpp
```

### Nvidia CUDA
In this version the `nvcc` Compiler from Nvidia is used. This makes use iof Nvidia GPUs. The
architecture options of the compiler have to match your GPU, so this needs some changes in
the makefile.

For compilation type
```
make cuda
```

--
If both versions are needed the make target `all` is your friend
```
make all
```

### Python
There is also a Python interface, originally this was only for testing purposes. It turnde out
that this interface is quite convenient and well suited for quick tests and prototyping.
There are also some support functions that make life easier.

For the python test inteface type
```
make python
```

Note
--
There are no binary data files for testing included, if these are
needed they have to be created by the user.

## License
MIT licensed, see [LICENSE](./LICENSE)
