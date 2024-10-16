
#include "HardeningModel.h"

InputParameters
HardeningModel::validParams()
{
  InputParameters params = Material::validParams();
  params.addParam<std::string>("base_name", "Material property base name");
  params.addParam<bool>(
      "auto_derivation", true, "Automatic derivation using finite difference method (true)");
  params.addParam<Real>("eps", 1e-9, " ");
  params.addRequiredParam<MaterialPropertyName>("sigma_0", "The initial yield stress");
  params.addRangeCheckedParam<Real>(
      "TQ",
      0,
      "TQ<=1 & TQ>=0",
      "The Taylor-Quinney factor. 1 for purely dissipative, 0 (default) for purely energetic.");
  params.set<bool>("compute") = false;
  params.suppressParameter<bool>("compute");
  return params;
}

HardeningModel::HardeningModel(const InputParameters & parameters)
  : Material(parameters),
    _base_name(isParamValid("base_name") ? getParam<std::string>("base_name") + "_" : ""),
    _auto_derivation(getParam<bool>("auto_derivation")),
    _eps(getParam<Real>("eps")),
    _sigma_0(getMaterialProperty<Real>("sigma_0")),
    _TQ(getParam<Real>("TQ")),
    _ep(getMaterialPropertyByName<Real>(_base_name + "effective_plastic_strain")),
    _ep_old(getMaterialPropertyOldByName<Real>(_base_name + "effective_plastic_strain")),
    _epdot(declareProperty<Real>(_base_name + "epdot")),
    _psip(declareProperty<Real>(_base_name + "psip")),
    _psip_old(getMaterialPropertyOldByName<Real>(_base_name + "psip")),
    _psip_active(declareProperty<Real>(_base_name + "psip_active")),
    _heat_generation_rate(declareProperty<Real>(_base_name + "plastic_heat_generation_rate")),
    _sigma(declareProperty<Real>(_base_name + "plastic_sigma")),
    _sigma_old(getMaterialPropertyOldByName<Real>(_base_name + "plastic_sigma")),
    _dsigma(declareProperty<Real>(_base_name + "plastic_dsigma")),
    _d2sigma(declareProperty<Real>(_base_name + "plastic_d2sigma")),
    _sigma_RI(declareProperty<Real>(_base_name + "plastic_sigma_RI")),
    _sigma_RI_old(getMaterialPropertyOldByName<Real>(_base_name + "plastic_sigma_RI"))
{
}

void
HardeningModel::initQpStatefulProperties()
{
  _psip[_qp] = 0.0;
  _psip_active[_qp] = 0.0;
  _heat_generation_rate[_qp] = 0.0;
  _sigma[_qp] = _sigma_0[_qp];
  _epdot[_qp] = 0.0;
  computeHardeningStressRI();
}

void
HardeningModel::resetDerivativesAtQp()
{
  _dsigma[_qp] = 0;
  _d2sigma[_qp] = 0;
}

void
HardeningModel::computeEnergy()
{
  // Plastic energy is calculated incrementally.
  // Notice: Make sure _sigma has been updated.
  Real plastic_work = (_sigma[_qp] + _sigma_old[_qp]) * (_ep[_qp] - _ep_old[_qp]) / 2;
  computeHardeningStressRI();
  Real plastic_work_RI = (_sigma_RI[_qp] + _sigma_RI_old[_qp]) * (_ep[_qp] - _ep_old[_qp]) / 2;
  _psip[_qp] = _psip_old[_qp] + (1.0 - _TQ) * plastic_work_RI;
  _psip_active[_qp] = _psip[_qp];
  _heat_generation_rate[_qp] =
      (plastic_work - (1.0 - _TQ) * plastic_work_RI) / _dt + thermalConjugate();
}

void
HardeningModel::computeHardeningStress()
{
  _sigma[_qp] = computeAnalyticalHardeningStress(_ep[_qp]);
}

void
HardeningModel::computeHardeningStressDerivative()
{
  if (_auto_derivation)
  {
    _dsigma[_qp] = (computeAnalyticalHardeningStress(_ep[_qp] + _eps) -
                    computeAnalyticalHardeningStress(_ep[_qp])) /
                   _eps;
  }
  else
  {
    _dsigma[_qp] = computeAnalyticalDerivative();
  }
}

void
HardeningModel::computeHardeningStressDerivative2()
{
  if (_auto_derivation)
  {
    _d2sigma[_qp] = (computeAnalyticalHardeningStress(_ep[_qp] + 2 * _eps) +
                     computeAnalyticalHardeningStress(_ep[_qp]) -
                     computeAnalyticalHardeningStress(_ep[_qp] + _eps) * 2) /
                    _eps / _eps;
  }
  else
  {
    _d2sigma[_qp] = computeAnalyticalDerivative2();
  }
}

Real
HardeningModel::computeAnalyticalDerivative()
{
  mooseError("computeAnalyticalDerivative must be overriden when auto_derivation=false");
  return 0.0;
}

Real
HardeningModel::computeAnalyticalDerivative2()
{
  mooseError("computeAnalyticalDerivative2 must be overriden when auto_derivation=false");
  return 0.0;
}