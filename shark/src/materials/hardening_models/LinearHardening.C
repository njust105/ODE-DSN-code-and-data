#include "LinearHardening.h"

registerMooseObject("sharkApp", LinearHardening);

InputParameters
LinearHardening::validParams()
{
  InputParameters params = HardeningModel::validParams();
  params.addClassDescription("Linear hardening model.");
  params.addRequiredParam<MaterialPropertyName>("hardening_modulus", "The hardening modulus $H$");
  return params;
}

LinearHardening::LinearHardening(const InputParameters & parameters)
  : HardeningModel(parameters), _H(getMaterialProperty<Real>("hardening_modulus"))
{
}

Real
LinearHardening::computeAnalyticalHardeningStress(const Real & ep)
{
  return _sigma_old[_qp] + _H[_qp] * (ep - _ep_old[_qp]);
}

Real
LinearHardening::computeAnalyticalDerivative()
{
  return _H[_qp];
}