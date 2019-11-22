#ifndef _CELERITE_TERMS_HPP_DEFINED_
#define _CELERITE_TERMS_HPP_DEFINED_

#include <tuple>
#include <exception>
#include <Eigen/Core>

namespace celerite {

struct dimension_mismatch : public std::exception {
  const char *what() const throw() { return "dimension mismatch"; }
};

template <typename T, int _J = Eigen::Dynamic>
class Term {
  public:
  typedef T Scalar;
  typedef Eigen::Matrix<T, Eigen::Dynamic, 1> Vector;
  typedef std::tuple<Vector, Vector, Vector, Vector, Vector, Vector> Coeffs;

  static constexpr int Width = _J;
  static constexpr int Order = (_J != 1) ? Eigen::RowMajor : Eigen::ColMajor;

  Term(){};

  virtual int get_J() const { return 0; };
  virtual Coeffs get_coefficients() const {
    Vector ar, cr, ac, bc, cc, dc;
    return std::make_tuple(ar, cr, ac, bc, cc, dc);
  };

  auto get_celerite_matrices(const Vector &x, const Vector &diag) const {
    int N = x.rows();
    if (diag.rows() != N) throw dimension_mismatch();

    int J = this->get_J();
    if (_J != Eigen::Dynamic && _J != J) throw dimension_mismatch();

    Vector ar, cr, ac, bc, cc, dc;
    std::tie(ar, cr, ac, bc, cc, dc) = this->get_coefficients();
    int nr                           = ar.rows();
    int nc                           = ac.rows();
    if (nr + 2 * nc != J) throw dimension_mismatch();

    Vector a = diag.array() + (ar.sum() + ac.sum());
    Eigen::Matrix<T, Eigen::Dynamic, _J, Order> U(N, J), V(N, J), P(N - 1, J);

    auto dx = x.head(N - 1) - x.segment(1, N - 1);

    U.block(0, 0, N, nr).rowwise() = ar.transpose();
    V.block(0, 0, N, nr).setConstant(T(1));
    P.block(0, 0, N - 1, nr) = exp(-(dx * cr.transpose()).array());

    auto arg                   = (x * dc.transpose()).array();
    auto ca                    = cos(arg);
    auto sa                    = sin(arg);
    U.block(0, nr, N, nc)      = ca.array().rowwise() * ac.transpose().array() + sa.array().rowwise() * bc.transpose().array();
    U.block(0, nr + nc, N, nc) = sa.array().rowwise() * ac.transpose().array() - ca.array().rowwise() * bc.transpose().array();
    V.block(0, nr, N, nc)      = ca;
    V.block(0, nr + nc, N, nc) = sa;
    P.block(0, nr, N - 1, nc) = P.block(0, nr + nc, N - 1, nc) = exp(-(dx * cc.transpose()).array());

    return std::make_tuple(a, U, V, P);
  };

  template <typename Other>
  friend auto operator+(Term<T, _J> const &, Other const &);
};

template <typename Term1, typename Term2>
constexpr int get_sum_width() {
  if (Term1::Width == Eigen::Dynamic || Term2::Width == Eigen::Dynamic) { return Eigen::Dynamic; }
  return Term1::Width + Term2::Width;
}

template <typename Term1, typename Term2>
class TermSum
   : public Term<decltype(std::declval<typename Term1::Scalar>() + std::declval<typename Term2::Scalar>()), get_sum_width<Term1, Term2>()> {
  public:
  typedef decltype(std::declval<typename Term1::Scalar>() + std::declval<typename Term2::Scalar>()) Scalar;
  constexpr static int Width = get_sum_width<Term1, Term2>();

  using Term<Scalar, Width>::Order;
  using typename Term<Scalar, Width>::Vector;
  using typename Term<Scalar, Width>::Coeffs;

  TermSum(const Term1 &term1, const Term2 term2) : Term<Scalar, Width>(), term1(term1), term2(term2){};

  int get_J() const { return term1.get_J() + term2.get_J(); };
  Coeffs get_coefficients() const {
    Vector ar1, cr1, ac1, bc1, cc1, dc1;
    std::tie(ar1, cr1, ac1, bc1, cc1, dc1) = term1.get_coefficients();
    Vector ar2, cr2, ac2, bc2, cc2, dc2;
    std::tie(ar2, cr2, ac2, bc2, cc2, dc2) = term2.get_coefficients();
    const int Jr = ar1.rows() + ar2.rows(), Jc = ac1.rows() + ac2.rows();
    Vector ar(Jr), cr(Jr), ac(Jc), bc(Jc), cc(Jc), dc(Jc);

    ar << ar1, ar2;
    cr << cr1, cr2;
    ac << ac1, ac2;
    bc << bc1, bc2;
    cc << cc1, cc2;
    dc << dc1, dc2;

    return std::make_tuple(ar, cr, ac, bc, cc, dc);
  };

  private:
  Term1 term1;
  Term2 term2;
};

template <typename Term1, typename Term2>
TermSum<Term1, Term2> operator+(const Term1 &term1, const Term2 &term2) {
  return TermSum<Term1, Term2>(term1, term2);
};

template <typename T>
class SHOTerm : public Term<T, 2> {
  public:
  using typename Term<T, 2>::Scalar;
  using typename Term<T, 2>::Vector;
  using typename Term<T, 2>::Coeffs;
  using Term<T, 2>::Width;
  using Term<T, 2>::Order;

  SHOTerm(T S0, T w0, T Q) : Term<T, 2>(), S0_(S0), w0_(w0), Q_(Q){};

  int get_J() const { return 2; }
  Coeffs get_coefficients() const {
    Vector ar, cr, ac, bc, cc, dc;

    if (Q_ < 0.5) {
      ar.resize(2);
      cr.resize(2);

      auto f = std::sqrt(std::max(1.0 - 4.0 * Q_ * Q_, 1e-5));
      auto a = 0.5 * S0_ * w0_ * Q_;
      auto c = 0.5 * w0_ / Q_;
      ar(0)  = a * (1 + 1 / f);
      ar(1)  = a * (1 - 1 / f);
      cr(0)  = c * (1 - f);
      cr(1)  = c * (1 + f);
    } else {
      ac.resize(1);
      bc.resize(1);
      cc.resize(1);
      dc.resize(1);

      auto f = std::sqrt(std::max(4.0 * Q_ * Q_ - 1, 1e-5));
      auto a = S0_ * w0_ * Q_;
      auto c = 0.5 * w0_ / Q_;
      ac(0)  = a;
      bc(0)  = a / f;
      cc(0)  = c;
      dc(0)  = c * f;
    }

    return std::make_tuple(ar, cr, ac, bc, cc, dc);
  };

  private:
  T S0_, w0_, Q_;
};

} // namespace celerite

#endif // _CELERITE_TERMS_HPP_DEFINED_