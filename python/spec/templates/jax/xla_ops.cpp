// Generated JAX FFI bindings for celerite2.
// Regenerate with: python python/spec/generate.py

#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <string>

#include "xla/ffi/api/ffi.h"

#include "../driver.hpp"

namespace py = pybind11;
namespace ffi = xla::ffi;
using namespace celerite2::driver;

// Helpers
template <int Axis, typename Buffer>
inline Eigen::Index dim(const Buffer& buf) {
  return static_cast<Eigen::Index>(buf.dimensions()[Axis]);
}
template <typename Buffer>
inline Eigen::Index flat_cols(const Buffer& buf) {
  const auto& dims = buf.dimensions();
  Eigen::Index cols = 1;
  for (size_t i = 1; i < dims.size(); ++i)
    cols *= static_cast<Eigen::Index>(dims[i]);
  return cols;
}

// === AUTO-GENERATED KERNELS ===
{% for mod in spec %}

ffi::Error {{mod.name|capitalize}}Impl(
{%- for arg in mod.inputs %}
    ffi::Buffer<ffi::DataType::F64> {{arg.name}}{% if not loop.last or mod.outputs or mod.extra_outputs %},{% endif %}
{%- endfor -%}
{%- for arg in mod.outputs %}
    ffi::ResultBuffer<ffi::DataType::F64> {{arg.name}}{% if not loop.last or mod.extra_outputs %},{% endif %}
{%- endfor -%}
{%- for arg in mod.extra_outputs %}
    ffi::ResultBuffer<ffi::DataType::F64> {{arg.name}}{% if not loop.last %},{% endif %}
{%- endfor %}
) {
  {# Dimension aliases for readability #}
  {% for dim in mod.dimensions %}
  const auto {{dim.name}} = dim<{{dim.coords[1]}}>({{mod.inputs[dim.coords[0]].name}});
  {% endfor %}
  {%- set nrhs_arg = None -%}
  {%- for a in mod.inputs + mod.outputs + mod.extra_outputs -%}
    {%- if a.shape|length >= 2 and a.shape[-1] == "nrhs" and nrhs_arg is none -%}
      {%- set nrhs_arg = a -%}
    {%- endif -%}
  {%- endfor -%}
  {%- if nrhs_arg is not none %}
  const auto nrhs = dim<{{ nrhs_arg.shape|length - 1 }}>({{ nrhs_arg.name }});
  {%- endif %}
  {# Minimal shape checks - rely on driver.hpp order helper #}
  {% for arg in mod.inputs %}
  {%- if arg.shape|length == 1 %}
  if (dim<0>({{arg.name}}) != {{arg.shape[0]}}) return ffi::Error::InvalidArgument("{{mod.name}} shape mismatch");
  {%- elif arg.shape|length == 2 %}
  if (dim<0>({{arg.name}}) != {{arg.shape[0]}} || dim<1>({{arg.name}}) != {{arg.shape[1]}}) return ffi::Error::InvalidArgument("{{mod.name}} shape mismatch");
  {%- endif %}
  {% endfor %}

#define FIXED_SIZE_MAP(SIZE) \
  { \
    {%- for arg in mod.inputs %}
    {%- if arg.shape|length == 1 %}
    Eigen::Map<const Eigen::VectorXd> {{arg.name}}_({{arg.name}}.typed_data(), {{arg.shape[0]}}, 1); \
    {%- elif arg.shape|length == 2 and arg.shape[1] == "J" %}
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> {{arg.name}}_({{arg.name}}.typed_data(), {{arg.shape[0]}}, J); \
    {%- elif arg.shape|length == 2 and arg.shape[1] == "nrhs" %}
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {{arg.name}}_({{arg.name}}.typed_data(), {{arg.shape[0]}}, dim<1>({{arg.name}})); \
    {%- else %}
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {{arg.name}}_({{arg.name}}.typed_data(), {{arg.shape[0]}}, dim<1>({{arg.name}})); \
    {%- endif %}
    {%- endfor %}
    {%- for arg in mod.outputs %}
    {%- if arg.shape|length == 1 %}
    Eigen::Map<Eigen::VectorXd> {{arg.name}}_({{arg.name}}->typed_data(), {{arg.shape[0]}}, 1); \
    {%- elif arg.shape|length == 2 and arg.shape[1] == "J" %}
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> {{arg.name}}_({{arg.name}}->typed_data(), {{arg.shape[0]}}, J); \
    {%- elif arg.shape|length == 2 and arg.shape[1] == "nrhs" %}
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {{arg.name}}_({{arg.name}}->typed_data(), {{arg.shape[0]}}, {{arg.shape[1]}}); \
    {%- else %}
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {{arg.name}}_({{arg.name}}->typed_data(), {{arg.shape[0]}}, {{arg.shape[1]}}); \
    {%- endif %}
    {%- endfor %}
    {%- for arg in mod.extra_outputs %}
    {%- if arg.shape|length == 3 and arg.shape[1] == "J" and arg.shape[2] == "J" %}
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, (SIZE * SIZE), order<(SIZE * SIZE)>::value>> {{arg.name}}_({{arg.name}}->typed_data(), {{arg.shape[0]}}, J * J); \
    {%- else %}
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {{arg.name}}_({{arg.name}}->typed_data(), {{arg.shape[0]}}, {{ '*'.join(arg.shape[1:]) }}); \
    {%- endif %}
    {%- endfor %}
    {%- for arg in mod.outputs + mod.extra_outputs %}
    {{arg.name}}_.setZero(); \
    {%- endfor %}
    try { \
        celerite2::core::{{mod.name}}({%- for arg in mod.inputs %} {{arg.name}}_{% if not loop.last or mod.outputs or mod.extra_outputs %},{% endif %}{%- endfor %}{%- if mod.outputs or mod.extra_outputs %} {% endif %}{%- for arg in mod.outputs + mod.extra_outputs %}{{arg.name}}_{% if not loop.last %},{% endif %}{%- endfor %}); \
    } catch (const std::exception& e) { \
        return ffi::Error::Internal(e.what()); \
    } \
  }
  UNWRAP_CASES_{{ "FEW" if mod.has_rev else "MOST" }}
#undef FIXED_SIZE_MAP

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {{mod.name}}, {{mod.name|capitalize}}Impl,
    ffi::Ffi::Bind()
{%- for arg in mod.inputs %}
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // {{arg.name}}
{%- endfor %}
{%- for arg in mod.outputs + mod.extra_outputs %}
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // {{arg.name}}
{%- endfor %}
);
{% if mod.has_rev %}

ffi::Error {{mod.name}}_revImpl(
{%- for arg in mod.rev_inputs %}
    ffi::Buffer<ffi::DataType::F64> {{arg.name}}{% if not loop.last or mod.rev_outputs %},{% endif %}
{%- endfor -%}
{%- for arg in mod.rev_outputs %}
    ffi::ResultBuffer<ffi::DataType::F64> {{arg.name}}{% if not loop.last %},{% endif %}
{%- endfor %}
) {
  {% for dim in mod.dimensions %}
  const auto {{dim.name}} = dim<{{dim.coords[1]}}>({{mod.inputs[dim.coords[0]].name}});
  {% endfor %}
  {# Minimal shape checks #}
  {% for arg in mod.rev_inputs %}
  {%- if arg.shape|length == 1 %}
  if (dim<0>({{arg.name}}) != {{arg.shape[0]}}) return ffi::Error::InvalidArgument("{{mod.name}}_rev shape mismatch");
  {%- elif arg.shape|length == 2 %}
  if (dim<0>({{arg.name}}) != {{arg.shape[0]}} || dim<1>({{arg.name}}) != {{arg.shape[1]}}) return ffi::Error::InvalidArgument("{{mod.name}}_rev shape mismatch");
  {%- endif %}
  {% endfor %}

#define FIXED_SIZE_MAP(SIZE) \
  { \
    {%- for arg in mod.rev_inputs %}
    {%- if arg.shape|length == 1 %}
    Eigen::Map<const Eigen::VectorXd> {{arg.name}}_({{arg.name}}.typed_data(), {{arg.shape[0]}}, 1); \
    {%- elif arg.shape|length == 2 and arg.shape[1] == "J" %}
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> {{arg.name}}_({{arg.name}}.typed_data(), {{arg.shape[0]}}, J); \
    {%- elif arg.shape|length == 3 and arg.shape[1] == "J" and arg.shape[2] == "J" %}
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, (SIZE * SIZE), order<(SIZE * SIZE)>::value>> {{arg.name}}_({{arg.name}}.typed_data(), {{arg.shape[0]}}, J * J); \
    {%- elif arg.shape|length == 3 %}
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {{arg.name}}_({{arg.name}}.typed_data(), {{arg.shape[0]}}, {{ '*'.join(arg.shape[1:]) }}); \
    {%- else %}
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {{arg.name}}_({{arg.name}}.typed_data(), {{arg.shape[0]}}, {{arg.shape[1]}}); \
    {%- endif %}
    {%- endfor %}
    {%- for arg in mod.rev_outputs %}
    {%- if arg.shape|length == 1 %}
    Eigen::Map<Eigen::VectorXd> {{arg.name}}_({{arg.name}}->typed_data(), {{arg.shape[0]}}, 1); \
    {%- elif arg.shape|length == 2 and arg.shape[1] == "J" %}
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> {{arg.name}}_({{arg.name}}->typed_data(), {{arg.shape[0]}}, J); \
    {%- else %}
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {{arg.name}}_({{arg.name}}->typed_data(), {{arg.shape[0]}}, {{arg.shape[1]}}); \
    {%- endif %}
    {%- endfor %}
    {%- for arg in mod.rev_outputs %}
    {{arg.name}}_.setZero(); \
    {%- endfor %}
    try { \
        celerite2::core::{{mod.name}}_rev({%- for arg in mod.rev_inputs %} {{arg.name}}_{% if not loop.last or mod.rev_outputs %},{% endif %}{%- endfor %}{%- if mod.rev_outputs %} {% endif %}{%- for arg in mod.rev_outputs %}{{arg.name}}_{% if not loop.last %},{% endif %}{%- endfor %}); \
    } catch (const std::exception& e) { \
        return ffi::Error::Internal(e.what()); \
    } \
  }
  UNWRAP_CASES_FEW
#undef FIXED_SIZE_MAP

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    {{mod.name}}_rev, {{mod.name}}_revImpl,
    ffi::Ffi::Bind()
{%- for arg in mod.rev_inputs %}
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // {{arg.name}}
{%- endfor %}
{%- for arg in mod.rev_outputs %}
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // {{arg.name}}
{%- endfor %}
);
{% endif %}
{% endfor %}

// Pybind --------------------------------------------------------------------
template <auto* Fn>
py::capsule Encapsulate() {
  return py::capsule(reinterpret_cast<void*>(Fn), "xla._CUSTOM_CALL_TARGET");
}

PYBIND11_MODULE(xla_ops, m) {
  {% for mod in spec %}
  m.def("{{mod.name}}", &Encapsulate<{{mod.name}}>);
  {%- if mod.has_rev %}
  m.def("{{mod.name}}_rev", &Encapsulate<{{mod.name}}_rev>);
  {%- endif %}
  {% endfor %}
}
