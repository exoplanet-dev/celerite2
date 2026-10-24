# XLA FFI headers

Header-only XLA FFI API (`xla/ffi/api/{api,c_api,ffi}.h`) used to build the
JAX extension (`python/celerite2/jax/xla_ops.cpp`), copied verbatim from
`jaxlib==0.8.0` (`jaxlib/include`). Licensed under Apache 2.0; see `LICENSE`.

Vendoring these means building celerite2 does not require installing jax. The
FFI ABI is versioned and newer jaxlib releases accept handlers compiled against
older headers, so these only need updating to use newer FFI features. To
update, copy the files from `jax.ffi.include_dir()` of the desired version.
