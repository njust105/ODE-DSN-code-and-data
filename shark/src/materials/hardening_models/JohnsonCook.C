#include "JohnsonCook.h"

registerMooseObject("sharkApp", JohnsonCook);

InputParameters
JohnsonCook::validParams()
{
  InputParameters params = JohnsonCookNoTemp::validParams();
  params.addClassDescription("Johnson-cook hardening model.");
  params.addRequiredCoupledVar("T", "Temperature");
  params.addRequiredParam<Real>("T0", "Reference temperature of the material.");
  params.addRequiredParam<Real>("Tm", "The melting temperature of the material.");
  params.addRequiredParam<Real>("m", "JC model parameter.");

  return params;
}

JohnsonCook::JohnsonCook(const InputParameters & parameters)
  : JohnsonCookNoTemp(parameters),
    _T(coupledValueOld("T")),
    _T0(getParam<Real>("T0")),
    _Tm(getParam<Real>("Tm")),
    _m(getParam<Real>("m"))
{
}

Real
JohnsonCook::computeAnalyticalHardeningStress(const Real & ep)
{
  return JohnsonCookNoTemp::computeAnalyticalHardeningStress(ep) * temperatureFactor();
}

Real
JohnsonCook::computeAnalyticalDerivative()
{
  return JohnsonCookNoTemp::computeAnalyticalDerivative() * temperatureFactor();
}

Real
JohnsonCook::computeAnalyticalDerivative2()
{
  return JohnsonCookNoTemp::computeAnalyticalDerivative2() * temperatureFactor();
}

Real
JohnsonCook::temperatureFactor()
{
  return 1.0 - std::pow((_T[_qp] - _T0) / (_Tm - _T0), _m);
}