#include <tuple>
#include <exception>
#include <Eigen/Core>

namespace celerite
{

struct dimension_mismatch : public std::exception
{
  const char *what() const throw()
  {
    return "dimension mismatch";
  }
};

template <typename T, int _J = Eigen::Dynamic>
class Term
{
public:
  const int Order = (_J != 1) ? Eigen::RowMajor : Eigen::ColMajor;

  Term(){};

  virtual int get_J() const { return 0; };
  virtual auto get_coefficients() const
  {
    Eigen::Matrix<T, Eigen::Dynamic, 1> ar, cr, ac, bc, cc, dc;
    return std::make_tuple(ar, cr, ac, bc, cc, dc);
  };

  auto get_celerite_matrices(const Eigen::Matrix<T, Eigen::Dynamic, 1> &x, const Eigen::Matrix<T, Eigen::Dynamic, 1> &diag) const
  {
    int N = x.rows();
    if (diag.rows() != N)
      throw dimension_mismatch();

    int J = this->get_J();
    if (_J != Eigen::Dynamic && _J != J)
      throw dimension_mismatch();

    auto [ar, cr, ac, bc, cc, dc] = this->get_coefficients();
    int nr = ar.rows();
    int nc = ac.rows();
    if (nr + 2 * nc != J)
      throw dimension_mismatch();

    Eigen::Matrix<T, Eigen::Dynamic, 1> a(N);
    Eigen::Matrix<T, Eigen::Dynamic, _J, Order> U(N, J), V(N, J), P(N - 1, J);

    auto dx = x.head(N - 1) - x.segment(1, N - 1);
    a = diag.array() + ar.sum() + ac.sum();

    U.block(0, 0, N, nr).array().rowwise() += ar;
    V.block(0, 0, N, nr).setConstant(T(1));
    P.block(0, 0, N - 1, nr) = exp(-cr * dx.array());

    auto arg = x * dc.transpose();
    auto ca = cos(arg);
    auto sa = sin(arg);
    U.block(0, nr, N, nc) = ca.array().rowwise() * ac.array() + sa.array().rowwise() * bc.array();
    U.block(0, nr + nc, N, nc) = sa.array().rowwise() * ac.array() - ca.array().rowwise() * bc.array();

    V.block(0, nr, N, nc) = ca;
    V.block(0, nr + nc, N, nc) = sa;

    P.block(0, nr, N - 1, nc) = P.block(0, nr + nc, N - 1, nc) = exp(-cc * dx.array());

    return std::make_tuple(a, U, V, P);
  };

private:
}

template <typename T, int _J = Eigen::Dynamic>
class SHOTerm : public Term<T, _J>
{
public:
  SHOTerm(T S0, T w0, T Q) : Term<T, _J>(), S0_(S0), w0_(w0), Q_(Q){};

  int get_J() const { return 2; }
  auto get_coefficients() const
  {
    Eigen::Matrix<T, Eigen::Dynamic, 1> ar, cr, ac, bc, cc, dc;
    return std::make_tuple(ar, cr, ac, bc, cc, dc);
  };

private:
  T S0_, w0_, Q_;
}

} // namespace celerite
