#include <pybind11/pybind11.h>
#include "../driver.hpp"

namespace py = pybind11;
using namespace celerite2::driver;

{% for mod in spec %}
auto {{mod.name}} (void *out_tuple, const void **in) {
    {% set input_index = mod.dimensions|length - 1 -%}
    void **out = reinterpret_cast<void **>(out_tuple);
    {%- for dim in mod.dimensions %}
    const int {{dim.name}} = *reinterpret_cast<const int *>(in[{{loop.index - 1}}]);
    {%- endfor %}
    {% for arg in mod.inputs %}
    const double *{{arg.name}} = reinterpret_cast<const double *>(in[{{ input_index + loop.index }}]);
    {%- endfor %}
    {%- for arg in mod.outputs + mod.extra_outputs %}
    double *{{arg.name}} = reinterpret_cast<double *>(out[{{ loop.index - 1 }}]);
    {%- endfor %}

#define FIXED_SIZE_MAP(SIZE) \
    { \
    {%- for arg in mod.inputs + mod.outputs + mod.extra_outputs %}
    {%- if arg.shape|length == 1 -%}
    {%- if arg.shape[0] == "J" %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, SIZE, 1>> {{arg.name}}_({{arg.name}}, J, 1); \
    {%- else %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::VectorXd> {{arg.name}}_({{arg.name}}, {{arg.shape[0]}}, 1); \
    {%- endif -%}
    {%- elif arg.shape|length == 2 -%}
    {%- if arg.shape[1] == "J" %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> {{arg.name}}_({{arg.name}}, {{arg.shape[0]}}, J); \
    {%- endif -%}
    {%- else -%}
    {%- if arg.shape[2] == "J" %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, (SIZE * SIZE), order<(SIZE * SIZE)>::value>> {{arg.name}}_({{arg.name}}, {{arg.shape[0]}}, J * J); \
    {%- endif -%}
    {%- endif -%}
    {% endfor %}
    {%- if mod.name == "factor" %}
    Eigen::Index flag = celerite2::core::{{mod.name}}({% for val in mod.inputs + mod.outputs + mod.extra_outputs %}{{val.name}}_{%- if not loop.last %}, {% endif %}{% endfor %}); \
    if (flag) d_.setZero(); \
    {%- else %}
    if (nrhs == 1) { \
        {% for arg in mod.inputs + mod.outputs + mod.extra_outputs %}
        {%- if arg.shape|length == 2 and arg.shape[1] == "nrhs" -%}
        Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::VectorXd> {{arg.name}}_({{arg.name}}, {{arg.shape[0]}}, 1); \
        {% elif arg.shape|length == 3 and arg.shape[2] == "nrhs" -%}
        Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> {{arg.name}}_({{arg.name}}, {{arg.shape[0]}}, J); \
        {% endif -%}
        {% endfor -%}
        {% for arg in mod.outputs -%}
        {{arg.name}}_.setZero(); \
        {% endfor -%}
        celerite2::core::{{mod.name}}({% for val in mod.inputs + mod.outputs + mod.extra_outputs %}{{val.name}}_{%- if not loop.last %}, {% endif %}{% endfor %}); \
    } else { \
        {% for arg in mod.inputs + mod.outputs + mod.extra_outputs %}
        {%- if (arg.shape|length == 2 and arg.shape[1] == "nrhs") or (arg.shape|length == 3 and arg.shape[2] == "nrhs") -%}
        Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {{arg.name}}_({{arg.name}}, {{arg.shape[0]}}, {{arg.shape[1:]|join(" * ")}}); \
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
}
{%- if mod.has_rev %}
auto {{mod.name}}_rev (void *out_tuple, const void **in) {
    void **out = reinterpret_cast<void **>(out_tuple);
    const int N = *reinterpret_cast<const int *>(in[0]);
    const int J = *reinterpret_cast<const int *>(in[1]);
    {%- if mod.name == "factor" -%}
    {%- set input_index = 1 -%}
    {%- else %}
    const int nrhs = *reinterpret_cast<const int *>(in[2]);
    {%- set input_index = 2 -%}
    {%- endif %}
    {% for arg in mod.rev_inputs %}
    const double *{{arg.name}} = reinterpret_cast<const double *>(in[{{ input_index + loop.index }}]);
    {%- endfor %}
    {%- for arg in mod.rev_outputs %}
    double *{{arg.name}} = reinterpret_cast<double *>(out[{{ loop.index - 1 }}]);
    {%- endfor %}

#define FIXED_SIZE_MAP(SIZE) \
    { \
    {%- for arg in mod.rev_inputs + mod.rev_outputs %}
    {%- if arg.shape|length == 1 -%}
    {%- if arg.shape[0] == "J" %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, SIZE, 1>> {{arg.name}}_({{arg.name}}, J, 1); \
    {%- else %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::VectorXd> {{arg.name}}_({{arg.name}}, {{arg.shape[0]}}, 1); \
    {%- endif -%}
    {%- elif arg.shape|length == 2 -%}
    {%- if arg.shape[1] == "J" %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> {{arg.name}}_({{arg.name}}, {{arg.shape[0]}}, J); \
    {%- endif -%}
    {%- else -%}
    {%- if arg.shape[2] == "J" %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, (SIZE * SIZE), order<(SIZE * SIZE)>::value>> {{arg.name}}_({{arg.name}}, {{arg.shape[0]}}, J * J); \
    {%- endif -%}
    {%- endif -%}
    {% endfor %}
    {%- if mod.name == "factor" %}
    celerite2::core::{{mod.name}}_rev({% for val in mod.rev_inputs + mod.rev_outputs %}{{val.name}}_{%- if not loop.last %}, {% endif %}{% endfor %}); \
    {%- else %}
    if (nrhs == 1) { \
        {% for arg in mod.rev_inputs + mod.rev_outputs %}
        {%- if arg.shape|length == 2 and arg.shape[1] == "nrhs" -%}
        Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::VectorXd> {{arg.name}}_({{arg.name}}, {{arg.shape[0]}}, 1); \
        {% elif arg.shape|length == 3 and arg.shape[2] == "nrhs" -%}
        Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> {{arg.name}}_({{arg.name}}, {{arg.shape[0]}}, J); \
        {% endif -%}
        {% endfor -%}
        celerite2::core::{{mod.name}}_rev({% for val in mod.rev_inputs + mod.rev_outputs %}{{val.name}}_{%- if not loop.last %}, {% endif %}{% endfor %}); \
    } else { \
        {% for arg in mod.rev_inputs + mod.rev_outputs %}
        {%- if (arg.shape|length == 2 and arg.shape[1] == "nrhs") or (arg.shape|length == 3 and arg.shape[2] == "nrhs") -%}
        Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {{arg.name}}_({{arg.name}}, {{arg.shape[0]}}, {{arg.shape[1:]|join(" * ")}}); \
        {% endif -%}
        {% endfor -%}
        celerite2::core::{{mod.name}}_rev({% for val in mod.rev_inputs + mod.rev_outputs %}{{val.name}}_{%- if not loop.last %}, {% endif %}{% endfor %}); \
    } \
    {%- endif %}
    }
    UNWRAP_CASES_FEW
#undef FIXED_SIZE_MAP
}
{% endif %}
{% endfor %}
PYBIND11_MODULE(xla_ops, m) {
    {%- for mod in spec %}
    m.def("{{mod.name}}", []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&{{mod.name}}, name);
    });
    {%- if mod.has_rev %}
    m.def("{{mod.name}}_rev", []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&{{mod.name}}_rev, name);
    });
    {%- endif %}
    {%- endfor %}
}
