#include <pybind11/pybind11.h>
#include "driver.hpp"

namespace py = pybind11;

namespace celerite2 {
namespace driver {
{% for mod in spec %}
auto {{mod.name}}_fwd (
    {% for arg in mod.inputs + mod.outputs + mod.extra_outputs -%}
    py::array_t<double, py::array::c_style> {{arg.name}}
    {%- if not loop.last %},
    {% endif %}
    {%- endfor %}
) {
    // Request buffers
    {% for arg in mod.inputs + mod.outputs + mod.extra_outputs -%}
    py::buffer_info {{arg.name}}buf = {{arg.name}}.request();
    {% endfor %}
    // Parse dimensions
    {% for dim in mod.dimensions -%}
    if ({{mod.inputs[dim.coords[0]].name}}buf.ndim <= {{dim.coords[1]}})
        throw std::invalid_argument("Invalid number of dimensions: {{mod.inputs[dim.coords[0]].name}}");
    ssize_t {{dim.name}} = {{mod.inputs[dim.coords[0]].name}}buf.shape[{{dim.coords[1]}}];
    {% endfor %}
    // Check shapes
    {% for arg in mod.inputs + mod.outputs + mod.extra_outputs -%}
    if ({{arg.name}}buf.ndim != {{arg.shape|length}}{% for dim in arg.shape %} || {{arg.name}}buf.shape[{{loop.index - 1}}] != {{dim}}{% endfor %}) throw std::invalid_argument("Invalid shape: {{arg.name}}");
    {% endfor %}
    {%- if mod.name == "factor" %}
    Eigen::Index flag = 0;{% endif %}
#define FIXED_SIZE_MAP(SIZE) \
    { \
    {%- for arg in mod.inputs + mod.outputs + mod.extra_outputs %}
    {%- if arg.shape|length == 1 -%}
    {%- if arg.shape[0] == "J" %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, SIZE, 1>> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, J, 1); \
    {%- else %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::VectorXd> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, {{arg.shape[0]}}, 1); \
    {%- endif -%}
    {%- elif arg.shape|length == 2 -%}
    {%- if arg.shape[1] == "J" %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, {{arg.shape[0]}}, J); \
    {%- endif -%}
    {%- else -%}
    {%- if arg.shape[2] == "J" %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, (SIZE * SIZE), order<(SIZE * SIZE)>::value>> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, {{arg.shape[0]}}, J * J); \
    {%- endif -%}
    {%- endif -%}
    {% endfor %}
    {%- if mod.name == "factor" %}
    flag = celerite2::core::{{mod.name}}({% for val in mod.inputs + mod.outputs + mod.extra_outputs %}{{val.name}}_{%- if not loop.last %}, {% endif %}{% endfor %}); \
    {%- else %}
    if (nrhs == 1) { \
        {% for arg in mod.inputs + mod.outputs + mod.extra_outputs %}
        {%- if arg.shape|length == 2 and arg.shape[1] == "nrhs" -%}
        Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::VectorXd> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, {{arg.shape[0]}}, 1); \
        {% elif arg.shape|length == 3 and arg.shape[2] == "nrhs" -%}
        Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, {{arg.shape[0]}}, J); \
        {% endif -%}
        {% endfor -%}
        {% for arg in mod.outputs -%}
        {{arg.name}}_.setZero(); \
        {% endfor -%}
        celerite2::core::{{mod.name}}({% for val in mod.inputs + mod.outputs + mod.extra_outputs %}{{val.name}}_{%- if not loop.last %}, {% endif %}{% endfor %}); \
    } else { \
        {% for arg in mod.inputs + mod.outputs + mod.extra_outputs %}
        {%- if (arg.shape|length == 2 and arg.shape[1] == "nrhs") or (arg.shape|length == 3 and arg.shape[2] == "nrhs") -%}
        Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, {{arg.shape[0]}}, {{arg.shape[1:]|join(" * ")}}); \
        {% endif -%}
        {% endfor -%}
        {% for arg in mod.outputs -%}
        {{arg.name}}_.setZero(); \
        {% endfor -%}
        celerite2::core::{{mod.name}}({% for val in mod.inputs + mod.outputs + mod.extra_outputs %}{{val.name}}_{%- if not loop.last %}, {% endif %}{% endfor %}); \
    } \
    {%- endif %}
    }
    UNWRAP_CASES_MOST
#undef FIXED_SIZE_MAP
    {% if mod.name == "factor" %}if (flag) throw backprop_linalg_exception();{% endif %}
    return std::make_tuple({{ (mod.outputs + mod.extra_outputs)|join(", ", attribute="name") }});
}
{%- if mod.has_rev %}
auto {{mod.name}}_rev (
    {% for arg in mod.rev_inputs + mod.rev_outputs -%}
    py::array_t<double, py::array::c_style> {{arg.name}}
    {%- if not loop.last %},
    {% endif %}
    {%- endfor %}
) {
    // Request buffers
    {% for arg in mod.rev_inputs + mod.rev_outputs -%}
    py::buffer_info {{arg.name}}buf = {{arg.name}}.request();
    {% endfor %}
    // Parse dimensions
    {% for dim in mod.dimensions -%}
    if ({{mod.inputs[dim.coords[0]].name}}buf.ndim <= {{dim.coords[1]}})
        throw std::invalid_argument("Invalid number of dimensions: {{mod.inputs[dim.coords[0]].name}}");
    ssize_t {{dim.name}} = {{mod.inputs[dim.coords[0]].name}}buf.shape[{{dim.coords[1]}}];
    {% endfor %}
    // Check shapes
    {% for arg in mod.rev_inputs + mod.rev_outputs -%}
    if ({{arg.name}}buf.ndim != {{arg.shape|length}}{% for dim in arg.shape %} || {{arg.name}}buf.shape[{{loop.index - 1}}] != {{dim}}{% endfor %}) throw std::invalid_argument("Invalid shape: {{arg.name}}");
    {% endfor %}
#define FIXED_SIZE_MAP(SIZE) \
    { \
    {%- for arg in mod.rev_inputs + mod.rev_outputs %}
    {%- if arg.shape|length == 1 -%}
    {%- if arg.shape[0] == "J" %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, SIZE, 1>> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, J, 1); \
    {%- else %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::VectorXd> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, {{arg.shape[0]}}, 1); \
    {%- endif -%}
    {%- elif arg.shape|length == 2 -%}
    {%- if arg.shape[1] == "J" %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, {{arg.shape[0]}}, J); \
    {%- endif -%}
    {%- else -%}
    {%- if arg.shape[2] == "J" %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, (SIZE * SIZE), order<(SIZE * SIZE)>::value>> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, {{arg.shape[0]}}, J * J); \
    {%- endif -%}
    {%- endif -%}
    {% endfor %}
    {%- if mod.name == "factor" %}
    celerite2::core::{{mod.name}}_rev({% for val in mod.rev_inputs + mod.rev_outputs %}{{val.name}}_{%- if not loop.last %}, {% endif %}{% endfor %}); \
    {%- else %}
    if (nrhs == 1) { \
        {% for arg in mod.rev_inputs + mod.rev_outputs %}
        {%- if arg.shape|length == 2 and arg.shape[1] == "nrhs" -%}
        Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::VectorXd> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, {{arg.shape[0]}}, 1); \
        {% elif arg.shape|length == 3 and arg.shape[2] == "nrhs" -%}
        Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, {{arg.shape[0]}}, J); \
        {% endif -%}
        {% endfor -%}
        celerite2::core::{{mod.name}}_rev({% for val in mod.rev_inputs + mod.rev_outputs %}{{val.name}}_{%- if not loop.last %}, {% endif %}{% endfor %}); \
    } else { \
        {% for arg in mod.rev_inputs + mod.rev_outputs %}
        {%- if (arg.shape|length == 2 and arg.shape[1] == "nrhs") or (arg.shape|length == 3 and arg.shape[2] == "nrhs") -%}
        Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, {{arg.shape[0]}}, {{arg.shape[1:]|join(" * ")}}); \
        {% endif -%}
        {% endfor -%}
        celerite2::core::{{mod.name}}_rev({% for val in mod.rev_inputs + mod.rev_outputs %}{{val.name}}_{%- if not loop.last %}, {% endif %}{% endfor %}); \
    } \
    {%- endif %}
    }
    UNWRAP_CASES_FEW
#undef FIXED_SIZE_MAP
    {% if mod.rev_outputs|length > 1 %}
    return std::make_tuple({{ mod.rev_outputs|join(", ", attribute="name") }});
    {%- else %}
    return {{mod.rev_outputs[0].name}};
    {%- endif %}
}
{% endif %}
{% endfor %}
} // namespace driver
} // namespace celerite2

PYBIND11_MODULE(backprop, m) {
    py::register_exception<celerite2::driver::backprop_linalg_exception>(m, "LinAlgError");

{%- for mod in spec %}
    m.def("{{mod.name}}_fwd", &celerite2::driver::{{mod.name}}_fwd);
    {%- if mod.has_rev %}
    m.def("{{mod.name}}_rev", &celerite2::driver::{{mod.name}}_rev);{% endif %}
{%- endfor %}

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
