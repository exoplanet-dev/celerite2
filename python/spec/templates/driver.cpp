#include <pybind11/pybind11.h>
#include <cmath>
#include "driver.hpp"

namespace py = pybind11;

namespace celerite2 {
namespace driver {
{% for mod in spec %}
auto {{mod.name}} (
    {% for arg in mod.inputs + mod.outputs -%}
    py::array_t<double, py::array::c_style> {{arg.name}}
    {%- if not loop.last %},
    {% endif %}
    {%- endfor %}
) {
    // Request buffers
    {% for arg in mod.inputs + mod.outputs -%}
    py::buffer_info {{arg.name}}buf = {{arg.name}}.request();
    {% endfor %}
    // Parse dimensions
    {% for dim in mod.dimensions -%}
    if ({{mod.inputs[dim.coords[0]].name}}buf.ndim <= {{dim.coords[1]}})
        throw std::invalid_argument("Invalid number of dimensions: {{mod.inputs[dim.coords[0]].name}}");
    ssize_t {{dim.name}} = {{mod.inputs[dim.coords[0]].name}}buf.shape[{{dim.coords[1]}}];
    {% endfor %}
    // Check shapes
    {% for arg in mod.inputs + mod.outputs -%}
    if ({{arg.name}}buf.ndim != {{arg.shape|length}}{% for dim in arg.shape %} || {{arg.name}}buf.shape[{{loop.index - 1}}] != {{dim}}{% endfor %}) throw std::invalid_argument("Invalid shape: {{arg.name}}");
    {% endfor %}
    {%- if mod.name == "factor" %}
    Eigen::Index flag = 0;{% endif %}
#define FIXED_SIZE_MAP(SIZE) \
    { \
    {%- for arg in mod.inputs + mod.outputs %}
    {%- if arg.shape|length == 1 -%}
    {%- if arg.shape[0] == "J" %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, SIZE, 1>> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, J, 1); \
    {%- else %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::VectorXd> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, {{arg.shape[0]}}, 1); \
    {%- endif -%}
    {%- else -%}
    {%- if arg.shape[1] == "J" %}
    Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, {{arg.shape[0]}}, J); \
    {%- endif -%}
    {%- endif -%}
    {% endfor %}
    {%- if mod.name == "factor" %}
    flag = celerite2::core::{{mod.name}}({% for val in mod.inputs + mod.outputs %}{{val.name}}_{%- if not loop.last %}, {% endif %}{% endfor %}); \
    {%- else %}
    if (nrhs == 1) { \
        {% for arg in mod.inputs + mod.outputs %}
        {%- if arg.shape|length == 2 and arg.shape[1] == "nrhs" -%}
        Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::VectorXd> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, {{arg.shape[0]}}, 1); \
        {% endif -%}
        {% endfor -%}
        celerite2::core::{{mod.name}}({% for val in mod.inputs + mod.outputs %}{{val.name}}_{%- if not loop.last %}, {% endif %}{% endfor %}); \
    } else { \
        {% for arg in mod.inputs + mod.outputs %}
        {%- if arg.shape|length == 2 and arg.shape[1] == "nrhs" -%}
        Eigen::Map<{% if not arg.is_output %}const {% endif %}Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> {{arg.name}}_(({% if not arg.is_output %}const {% endif %}double *){{arg.name}}buf.ptr, {{arg.shape[0]}}, nrhs); \
        {% endif -%}
        {% endfor -%}
        celerite2::core::{{mod.name}}({% for val in mod.inputs + mod.outputs %}{{val.name}}_{%- if not loop.last %}, {% endif %}{% endfor %}); \
    } \
    {%- endif %}
    }
    UNWRAP_CASES_MOST
#undef FIXED_SIZE_MAP
    {% if mod.name == "factor" %}if (flag) throw driver_linalg_exception();{% endif -%}
    {% if mod.outputs|length > 1 %}
    return std::make_tuple({{ mod.outputs|join(", ", attribute="name") }});
    {%- else %}
    return {{mod.outputs[0].name}};
    {%- endif %}
}
{% endfor %}
auto get_celerite_matrices(
    py::array_t<double, py::array::c_style> ar_in,
    py::array_t<double, py::array::c_style> ac_in,
    py::array_t<double, py::array::c_style> bc_in,
    py::array_t<double, py::array::c_style> dc_in,
    py::array_t<double, py::array::c_style> x_in,
    py::array_t<double, py::array::c_style> diag_in,
    py::array_t<double, py::array::c_style> a_out,
    py::array_t<double, py::array::c_style> U_out,
    py::array_t<double, py::array::c_style> V_out
) {
    auto ar = ar_in.unchecked<1>();
    auto ac = ac_in.unchecked<1>();
    auto bc = bc_in.unchecked<1>();
    auto dc = dc_in.unchecked<1>();

    auto x    = x_in.unchecked<1>();
    auto diag = diag_in.unchecked<1>();

    auto a = a_out.mutable_unchecked<1>();
    auto U = U_out.mutable_unchecked<2>();
    auto V = V_out.mutable_unchecked<2>();

    ssize_t N = x.shape(0), Jr = ar.shape(0), Jc = ac.shape(0), J = Jr + 2 * Jc;

    if (bc.shape(0) != Jc) throw std::invalid_argument("dimension mismatch: bc");
    if (dc.shape(0) != Jc) throw std::invalid_argument("dimension mismatch: dc");

    if (diag.shape(0) != N) throw std::invalid_argument("dimension mismatch: diag");

    if (a.shape(0) != N) throw std::invalid_argument("dimension mismatch: a");
    if (U.shape(0) != N || U.shape(1) != J) throw std::invalid_argument("dimension mismatch: U");
    if (V.shape(0) != N || V.shape(1) != J) throw std::invalid_argument("dimension mismatch: V");

    double sum = 0.0;
    for (ssize_t j = 0; j < Jr; ++j) sum += ar(j);
    for (ssize_t j = 0; j < Jc; ++j) sum += ac(j);

    for (ssize_t n = 0; n < N; ++n) {
        a(n) = diag(n) + sum;
        for (ssize_t j = 0; j < Jr; ++j) {
            V(n, j) = 1.0;
            U(n, j) = ar(j);
        }
        for (ssize_t j = 0, ind = Jr; j < Jc; ++j, ind += 2) {
            double arg = dc(j) * x(n);
            double cos = V(n, ind) = std::cos(arg);
            double sin = V(n, ind + 1) = std::sin(arg);

            U(n, ind)     = ac(j) * cos + bc(j) * sin;
            U(n, ind + 1) = ac(j) * sin - bc(j) * cos;
        }
    }

    return std::make_tuple(a_out, U_out, V_out);
}

} // namespace driver
} // namespace celerite2

PYBIND11_MODULE(driver, m) {
    py::register_exception<celerite2::driver::driver_linalg_exception>(m, "LinAlgError");

{%- for mod in spec %}
    m.def("{{mod.name}}", &celerite2::driver::{{mod.name}});
{%- endfor %}

    m.def("get_celerite_matrices", &celerite2::driver::get_celerite_matrices);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
