#include "JohnsonCookNoTemp.h"

registerMooseObject("sharkApp", JohnsonCookNoTemp);

InputParameters
JohnsonCookNoTemp::validParams()
{
  InputParameters params = HardeningModel::validParams();
  params.addClassDescription("Johnson-cook hardening model.");
  params.addRequiredParam<Real>("B", "JC model parameter.");
  params.addRequiredParam<Real>("C", "JC model parameter.");
  params.addRequiredParam<Real>("n", "JC model parameter.");
  params.addRequiredParam<Real>("epdot0",
                                "The reference plastic strain rate parameter in the JC model.");

  return params;
}

JohnsonCookNoTemp::JohnsonCookNoTemp(const InputParameters & parameters)
  : HardeningModel(parameters),
    _B(getParam<Real>("B")),
    _C(getParam<Real>("C")),
    _n(getParam<Real>("n")),
    _epdot0(getParam<Real>("epdot0")),
    _epdot_old(getMaterialPropertyOlderByName<Real>(_base_name + "epdot"))
{
}

Real
JohnsonCookNoTemp::initialGuess(const Real & effective_trial_stress)
{
  return _auto_derivation ? _ep_old[_qp] : std::max(_ep_old[_qp], _eps);
  // return _auto_derivation ? 0.0 : libMesh::TOLERANCE * libMesh::TOLERANCE;
}

Real
JohnsonCookNoTemp::computeAnalyticalHardeningStress(const Real & ep)
{
  return (_sigma_0[_qp] + _B * std::pow(ep, _n)) * rateFactor();
}

Real
JohnsonCookNoTemp::computeAnalyticalDerivative()
{
  return (_B * std::pow(_ep[_qp], _n - 1) * _n) * rateFactor();
}

Real
JohnsonCookNoTemp::computeAnalyticalDerivative2()
{
  return (_B * std::pow(_ep[_qp], _n - 2) * _n * (_n - 1)) * rateFactor();
}

Real
JohnsonCookNoTemp::rateFactor()
{
  return 1.0 + _C * std::log(std::max(_epdot_old[_qp] / _epdot0, 1.0));
}

void
JohnsonCookNoTemp::computeHardeningStressRI()
{
  _sigma_RI[_qp] = (_sigma_0[_qp] + _B * std::pow(_ep[_qp], _n));
}