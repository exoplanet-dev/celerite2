#ifndef _CELERITE2_TERMS_HPP_DEFINED_
#define _CELERITE2_TERMS_HPP_DEFINED_

#include <tuple>
#include <exception>
#include <Eigen/Core>

namespace celerite2 {

#ifndef CELERITE_MAX_WIDTH
#define CELERITE_MAX_WIDTH 32
#endif

struct dimension_mismatch : public std::exception {
  const char *what() const throw() { return "dimension mismatch"; }
};

template <int J1, int J2>
struct sum_width {
  constexpr static int value = (J1 == Eigen::Dynamic || J2 == Eigen::Dynamic) ? Eigen::Dynamic : (J1 + J2);
};

template <typename T, int J_ = Eigen::Dynamic>
class Term {
  public:
  typedef T Scalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  constexpr static int Width = ((0 < J_) && (J_ <= CELERITE_MAX_WIDTH)) ? J_ : Eigen::Dynamic;
  static constexpr int Order = (Width != 1) ? Eigen::RowMajor : Eigen::ColMajor;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Width, Order> LowRank;
  typedef std::tuple<Vector, LowRank, LowRank, LowRank> Matrices;
  typedef std::tuple<Vector, Vector, Vector, Vector, Vector, Vector> Coeffs;

  Term(){};

  void set_coefficients(const Vector &ar, const Vector &cr, const Vector &ac, const Vector &bc, const Vector &cc, const Vector &dc) {
    Eigen::Index nr = ar.rows(), nc = ac.rows();

    ar_.resize(nr);
    cr_.resize(nr);
    ac_.resize(nc);
    bc_.resize(nc);
    cc_.resize(nc);
    dc_.resize(nc);

    ar_ << ar;
    cr_ << cr;
    ac_ << ac;
    bc_ << bc;
    cc_ << cc;
    dc_ << dc;
  }

  Coeffs get_coefficients() const { return std::make_tuple(ar_, cr_, ac_, bc_, cc_, dc_); }

  Matrices get_celerite_matrices(const Vector &x, const Vector &diag) const {
    Eigen::Index N = x.rows();
    if (diag.rows() != N) throw dimension_mismatch();

    Eigen::Index nr = ar_.rows();
    Eigen::Index nc = ac_.rows();
    Eigen::Index J  = nr + 2 * nc;
    if (Width != Eigen::Dynamic && Width != J) throw dimension_mismatch();

    Vector a = diag.array() + (ar_.sum() + ac_.sum());
    LowRank U(N, J), V(N, J), P(N - 1, J);

    Vector dx = x.segment(1, N - 1) - x.head(N - 1);

    U.block(0, 0, N, nr).rowwise() = ar_.transpose();
    V.block(0, 0, N, nr).setConstant(Scalar(1));
    P.block(0, 0, N - 1, nr) = exp(-(dx * cr_.transpose()).array());

    auto arg                   = (x * dc_.transpose()).array().eval();
    auto ca                    = cos(arg).eval();
    auto sa                    = sin(arg).eval();
    U.block(0, nr, N, nc)      = ca.array().rowwise() * ac_.transpose().array() + sa.array().rowwise() * bc_.transpose().array();
    U.block(0, nr + nc, N, nc) = sa.array().rowwise() * ac_.transpose().array() - ca.array().rowwise() * bc_.transpose().array();
    V.block(0, nr, N, nc)      = ca;
    V.block(0, nr + nc, N, nc) = sa;
    P.block(0, nr, N - 1, nc) = P.block(0, nr + nc, N - 1, nc) = exp(-(dx * cc_.transpose()).array());

    return std::make_tuple(a, U, V, P);
  }

  template <typename Other>
  Term<typename std::common_type<Scalar, typename Other::Scalar>::type, sum_width<Width, Other::Width>::value> operator+(const Other &other) const {
    typedef typename std::common_type<Scalar, typename Other::Scalar>::type NewScalar;
    // constexpr int NewWidth = sum_width<Width, Other::Width>;

    auto coeffs = other.get_coefficients();

    Eigen::Index nr = ar_.rows() + std::get<0>(coeffs).rows();
    Eigen::Index nc = ac_.rows() + std::get<2>(coeffs).rows();

    Eigen::Matrix<NewScalar, Eigen::Dynamic, 1> ar(nr), cr(nr), ac(nc), bc(nc), cc(nc), dc(nc);

    ar << ar_, std::get<0>(coeffs);
    cr << cr_, std::get<1>(coeffs);
    ac << ac_, std::get<2>(coeffs);
    bc << ac_, std::get<3>(coeffs);
    cc << ac_, std::get<4>(coeffs);
    dc << ac_, std::get<5>(coeffs);

    Term<NewScalar, sum_width<Width, Other::Width>::value> new_term;
    new_term.set_coefficients(ar, cr, ac, bc, cc, dc);

    return new_term;
  }

  private:
  Vector ar_, cr_, ac_, bc_, cc_, dc_;
};

template <typename T>
class RealTerm : public Term<T, 1> {
  public:
  typedef T Scalar;
  constexpr static int Width = 1;
  using typename Term<Scalar, 1>::Vector;
  using typename Term<Scalar, 1>::LowRank;
  RealTerm(const Scalar &a, const Scalar &c) {
    Vector ar(1), cr(1), ac, bc, cc, dc;
    ar << a;
    cr << c;
    this->set_coefficients(ar, cr, ac, bc, cc, dc);
  };
};

template <typename T>
class ComplexTerm : public Term<T, 2> {
  public:
  typedef T Scalar;
  constexpr static int Width = 2;
  using typename Term<Scalar, 2>::Vector;
  using typename Term<Scalar, 2>::LowRank;
  ComplexTerm(const Scalar &a, const Scalar &b, const Scalar &c, const Scalar &d) {
    Vector ar, cr, ac(1), bc(1), cc(1), dc(1);
    ac << a;
    bc << b;
    cc << c;
    dc << d;
    this->set_coefficients(ar, cr, ac, bc, cc, dc);
  };
};

template <typename T>
class SHOTerm : public Term<T, 2> {
  public:
  typedef T Scalar;
  constexpr static int Width = 2;
  using typename Term<Scalar, 2>::Vector;
  using typename Term<Scalar, 2>::LowRank;
  SHOTerm(const Scalar &S0, const Scalar &w0, const Scalar &Q, const Scalar &eps = 1e-5) {
    Vector ar, cr, ac, bc, cc, dc;
    if (Q < 0.5) {
      ar.resize(2);
      cr.resize(2);
      auto f = std::sqrt(std::max(1.0 - 4.0 * Q * Q, eps));
      auto a = 0.5 * S0 * w0 * Q;
      auto c = 0.5 * w0 / Q;
      ar(0)  = a * (1 + 1 / f);
      ar(1)  = a * (1 - 1 / f);
      cr(0)  = c * (1 - f);
      cr(1)  = c * (1 + f);
    } else {
      ac.resize(1);
      bc.resize(1);
      cc.resize(1);
      dc.resize(1);
      auto f = std::sqrt(std::max(4.0 * Q * Q - 1, eps));
      auto a = S0 * w0 * Q;
      auto c = 0.5 * w0 / Q;
      ac(0)  = a;
      bc(0)  = a / f;
      cc(0)  = c;
      dc(0)  = c * f;
    }
    this->set_coefficients(ar, cr, ac, bc, cc, dc);
  };
};

} // namespace celerite2

#endif // _CELERITE2_TERMS_HPP_DEFINED_
