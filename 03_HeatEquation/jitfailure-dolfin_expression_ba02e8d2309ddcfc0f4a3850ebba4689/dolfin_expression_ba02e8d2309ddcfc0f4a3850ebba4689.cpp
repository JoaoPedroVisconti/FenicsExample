
// Based on https://gcc.gnu.org/wiki/Visibility
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #define DLL_EXPORT __attribute__ ((dllexport))
    #else
        #define DLL_EXPORT __declspec(dllexport)
    #endif
#else
    #define DLL_EXPORT __attribute__ ((visibility ("default")))
#endif

#include <dolfin/function/Expression.h>
#include <dolfin/math/basic.h>
#include <Eigen/Dense>


// cmath functions
using std::cos;
using std::sin;
using std::tan;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::cosh;
using std::sinh;
using std::tanh;
using std::exp;
using std::frexp;
using std::ldexp;
using std::log;
using std::log10;
using std::modf;
using std::pow;
using std::sqrt;
using std::ceil;
using std::fabs;
using std::floor;
using std::fmod;
using std::max;
using std::min;

const double pi = DOLFIN_PI;


namespace dolfin
{
  class dolfin_expression_ba02e8d2309ddcfc0f4a3850ebba4689 : public Expression
  {
     public:
       double b;
double H;
double d_x;
double d_y;
double t;


       dolfin_expression_ba02e8d2309ddcfc0f4a3850ebba4689()
       {
            
       }

       void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x) const override
       {
          values[0] = 50*exp(-2*(pow((x[0]-t),2) + pow((x[1]-b*H),2) + pow((x[2]-b*H),2))/(d_x*d_x + d_y*d_y + d_z*d_z));

       }

       void set_property(std::string name, double _value) override
       {
          if (name == "b") { b = _value; return; }          if (name == "H") { H = _value; return; }          if (name == "d_x") { d_x = _value; return; }          if (name == "d_y") { d_y = _value; return; }          if (name == "t") { t = _value; return; }
       throw std::runtime_error("No such property");
       }

       double get_property(std::string name) const override
       {
          if (name == "b") return b;          if (name == "H") return H;          if (name == "d_x") return d_x;          if (name == "d_y") return d_y;          if (name == "t") return t;
       throw std::runtime_error("No such property");
       return 0.0;
       }

       void set_generic_function(std::string name, std::shared_ptr<dolfin::GenericFunction> _value) override
       {

       throw std::runtime_error("No such property");
       }

       std::shared_ptr<dolfin::GenericFunction> get_generic_function(std::string name) const override
       {

       throw std::runtime_error("No such property");
       }

  };
}

extern "C" DLL_EXPORT dolfin::Expression * create_dolfin_expression_ba02e8d2309ddcfc0f4a3850ebba4689()
{
  return new dolfin::dolfin_expression_ba02e8d2309ddcfc0f4a3850ebba4689;
}

