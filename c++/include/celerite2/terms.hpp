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

/**
 * The abstract base class from which terms should inherit
 */
template <typename T, int J_ = Eigen::Dynamic>
class Term {
  protected:
  constexpr static int Width = ((0 < J_) && (J_ <= CELERITE_MAX_WIDTH)) ? J_ : Eigen::Dynamic;
  static constexpr int Order = (Width != 1) ? Eigen::RowMajor : Eigen::ColMajor;

  public:
  /**
   * \typedef Scalar
   * The underlying scalar type of this `Term` (should probably always be `double`)
   */
  typedef T Scalar;

  /**
   * \typedef Vector
   * An `Eigen` vector with data type `Scalar`
   */
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

  /**
   * \typedef LowRank
   * The `Eigen` type for the low-rank matrices used internally
   */
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Width, Order> LowRank;

  /**
   * \typedef Coeffs
   * A tuple of vectors giving the coefficients for the celerite model
   */
  typedef std::tuple<Vector, Vector, Vector, Vector, Vector, Vector> Coeffs;

  /**
   * \typedef Matrices
   * A tuple of matrices representing this celerite process
   */
  typedef std::tuple<Vector, LowRank, LowRank, LowRank> Matrices;

  Term(){};

  int get_width() const { return Width; }

  /**
   * Set the coefficients of the term
   *
   * @param ar     (J_real,): The real amplitudes.
   * @param cr     (J_real,): The real exponential.
   * @param ac     (J_comp,): The complex even amplitude.
   * @param bc     (J_comp,): The complex odd amplitude.
   * @param cc     (J_comp,): The complex exponential.
   * @param dc     (J_comp,): The complex frequency.
   */
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

  /**
   * Get the coefficients of the term as a tuple
   */
  Coeffs get_coefficients() const { return std::make_tuple(ar_, cr_, ac_, bc_, cc_, dc_); }

  /**
   * Get the matrices required to represent the celerite process
   *
   * @param x    (N,): The independent coordinates of the data.
   * @param diag (N,): The diagonal variance of the process.
   */
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

  /**
   * Adding two terms builds a new term where the coefficients have been concatenated
   *
   * @param other (Term): The term to add to this one.
   */
  template <typename Other>
  Term<typename std::common_type<Scalar, typename Other::Scalar>::type, sum_width<Width, Other::Width>::value> operator+(const Other &other) const {
    typedef typename std::common_type<Scalar, typename Other::Scalar>::type NewScalar;

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

/**
 * \class RealTerm
 * The simplest celerite model
 *
 * @param a: The amplitude of the term.
 * @param c: The exponent of the term.
 */
template <typename T>
class RealTerm : public Term<T, 1> {
  public:
  /**
   * \typedef Scalar
   * The underlying scalar type of this `Term` (should probably always be `double`)
   */
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

/**
 * \class ComplexTerm
 * A general celerite model
 *
 * @param a: The real part of the amplitude.
 * @param b: The complex part of the amplitude.
 * @param c: The real part of the exponent.
 * @param d: The complex part of the exponent.
 */
template <typename T>
class ComplexTerm : public Term<T, 2> {
  public:
  /**
   * \typedef Scalar
   * The underlying scalar type of this `Term` (should probably always be `double`)
   */
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

/**
 * \class SHOTerm
 * A term representing a stochastically-driven, damped harmonic oscillator
 *
 * @param S0:  The power at `omega = 0`.
 * @param w0:  The undamped angular frequency.
 * @param Q:   The quality factor.
 * @param eps: A regularization parameter used for numerical stability.
 */
template <typename T>
class SHOTerm : public Term<T, 2> {
  public:
  /**
   * \typedef Scalar
   * The underlying scalar type of this `Term` (should probably always be `double`)
   */
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
