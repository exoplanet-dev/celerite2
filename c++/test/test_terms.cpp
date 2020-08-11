#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"

#include <celerite2/terms.hpp>

TEST_CASE("check the coefficients for RealTerm", "[terms]") {
  double a = 1.3, c = 0.1;
  celerite2::RealTerm<double> kernel(a, c);
  int width   = kernel.get_width();
  auto coeffs = kernel.get_coefficients();
  auto ar     = std::get<0>(coeffs);
  auto cr     = std::get<1>(coeffs);
  REQUIRE(ar.rows() == 1);
  REQUIRE(ar(0) == a);
  REQUIRE(cr.rows() == 1);
  REQUIRE(cr(0) == c);
  REQUIRE(width == 1);
}

TEST_CASE("check the coefficients for ComplexTerm", "[terms]") {
  double a = 1.3, b = 0.5, c = 0.1, d = 0.05;
  celerite2::ComplexTerm<double> kernel(a, b, c, d);
  int width   = kernel.get_width();
  auto coeffs = kernel.get_coefficients();
  auto ac     = std::get<2>(coeffs);
  auto bc     = std::get<3>(coeffs);
  auto cc     = std::get<4>(coeffs);
  auto dc     = std::get<5>(coeffs);
  REQUIRE(ac.rows() == 1);
  REQUIRE(ac(0) == a);
  REQUIRE(bc.rows() == 1);
  REQUIRE(bc(0) == b);
  REQUIRE(cc.rows() == 1);
  REQUIRE(cc(0) == c);
  REQUIRE(dc.rows() == 1);
  REQUIRE(dc(0) == d);
  REQUIRE(width == 2);
}

TEST_CASE("check the coefficients for SHOTerm", "[terms]") {
  double S0 = 1.5, w0 = 0.1, Q = 2.3;
  celerite2::SHOTerm<double> kernel(S0, w0, Q);
  int width   = kernel.get_width();
  auto coeffs = kernel.get_coefficients();
  auto ac     = std::get<2>(coeffs);
  auto bc     = std::get<3>(coeffs);
  auto cc     = std::get<4>(coeffs);
  auto dc     = std::get<5>(coeffs);
  REQUIRE(ac.rows() == 1);
  REQUIRE(bc.rows() == 1);
  REQUIRE(cc.rows() == 1);
  REQUIRE(dc.rows() == 1);
  REQUIRE(width == 2);
}

TEST_CASE("check the coefficients for a sum of terms", "[terms]") {
  double a0 = 0.45, c0 = 0.324, a = 1.3, b = 0.5, c = 0.1, d = 0.05;
  celerite2::RealTerm<double> term1(a0, c0);
  celerite2::ComplexTerm<double> term2(a, b, c, d);
  auto kernel = term1 + term2;
  int width   = kernel.get_width();
  auto coeffs = kernel.get_coefficients();
  auto ar     = std::get<0>(coeffs);
  auto cr     = std::get<1>(coeffs);
  auto ac     = std::get<2>(coeffs);
  auto bc     = std::get<3>(coeffs);
  auto cc     = std::get<4>(coeffs);
  auto dc     = std::get<5>(coeffs);
  REQUIRE(ar.rows() == 1);
  REQUIRE(ar(0) == a0);
  REQUIRE(cr.rows() == 1);
  REQUIRE(cr(0) == c0);
  REQUIRE(ac.rows() == 1);
  REQUIRE(ac(0) == a);
  REQUIRE(bc.rows() == 1);
  REQUIRE(bc(0) == b);
  REQUIRE(cc.rows() == 1);
  REQUIRE(cc(0) == c);
  REQUIRE(dc.rows() == 1);
  REQUIRE(dc(0) == d);
  REQUIRE(width == 3);
}
