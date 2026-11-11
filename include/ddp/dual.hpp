#pragma once
#include <cmath>
#include <type_traits>

// Forward-mode automatic differentiation with dual numbers a + b*eps, eps^2 = 0.
// Dual<double> carries one directional derivative. Dual<Dual<double>> carries two, and its
// .d.d component is the mixed second derivative, which is how DDP gets its second-order
// dynamics terms without hand-derived Hessians or finite differences.
namespace ddp {

template <typename T>
struct Dual {
  T v{};  // value
  T d{};  // derivative along the seeded direction
  Dual() = default;
  Dual(double x) : v(x), d(0.0) {}  // NOLINT: implicit from constants is the point
  Dual(const T& v_, const T& d_) : v(v_), d(d_) {}
  template <typename U = T, typename = std::enable_if_t<!std::is_same_v<U, double>>>
  Dual(const T& v_) : v(v_), d(0.0) {}  // NOLINT

  Dual& operator+=(const Dual& o) { v += o.v; d += o.d; return *this; }
  Dual& operator-=(const Dual& o) { v -= o.v; d -= o.d; return *this; }
  Dual& operator*=(const Dual& o) { d = d * o.v + v * o.d; v *= o.v; return *this; }
  Dual& operator/=(const Dual& o) { *this = *this / o; return *this; }
  friend Dual operator+(Dual a, const Dual& b) { return a += b; }
  friend Dual operator-(Dual a, const Dual& b) { return a -= b; }
  friend Dual operator-(const Dual& a) { return {-a.v, -a.d}; }
  friend Dual operator*(const Dual& a, const Dual& b) { return {a.v * b.v, a.d * b.v + a.v * b.d}; }
  friend Dual operator/(const Dual& a, const Dual& b) {
    return {a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v)};
  }
  friend bool operator<(const Dual& a, const Dual& b) { return a.v < b.v; }
  friend bool operator>(const Dual& a, const Dual& b) { return a.v > b.v; }
};

template <typename T> Dual<T> sin(const Dual<T>& a) { using std::sin; using std::cos; return {sin(a.v), a.d * cos(a.v)}; }
template <typename T> Dual<T> cos(const Dual<T>& a) { using std::sin; using std::cos; return {cos(a.v), -(a.d * sin(a.v))}; }
template <typename T> Dual<T> exp(const Dual<T>& a) { using std::exp; const T e = exp(a.v); return {e, a.d * e}; }
template <typename T> Dual<T> sqrt(const Dual<T>& a) { using std::sqrt; const T s = sqrt(a.v); return {s, a.d / (s * 2.0)}; }
template <typename T> Dual<T> tanh(const Dual<T>& a) {
  using std::tanh;
  const T t = tanh(a.v);
  return {t, a.d * (1.0 - t * t)};
}

// Plain value of a (possibly nested) dual, for logging and comparisons.
inline double value(double x) { return x; }
template <typename T> double value(const Dual<T>& a) { return value(a.v); }

}  // namespace ddp
