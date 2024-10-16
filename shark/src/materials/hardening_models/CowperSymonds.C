#include "CowperSymonds.h"

registerMooseObject("sharkApp", CowperSymonds);

InputParameters
CowperSymonds::validParams()
{
  InputParameters params = HardeningModel::validParams();
  params.addClassDescription("Cowper-Symonds model.");
  params.addRequiredParam<Real>("C", "Cowper-Symonds parameter.");
  params.addRequiredParam<Real>("P", "Cowper-Symonds parameter.");

  return params;
}

CowperSymonds::CowperSymonds(const InputParameters & parameters)
  : HardeningModel(parameters), _C(getParam<Real>("C")), _P(getParam<Real>("P"))
{
}

Real
CowperSymonds::initialGuess(const Real & effective_trial_stress)
{
  return _ep_old[_qp] + _dt * _C * std::pow(effective_trial_stress / _sigma_0[_qp] - 1, _P);
}

Real
CowperSymonds::computeAnalyticalHardeningStress(const Real & ep)
{
  Real epdot = std::max((ep - _ep_old[_qp]) / _dt, 0.0);
  return _sigma_0[_qp] * (1.0 + std::pow(epdot / _C, 1.0 / _P));
}
