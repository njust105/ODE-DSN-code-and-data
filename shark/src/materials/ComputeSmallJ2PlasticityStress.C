// #ifdef DONOTCOMPLIE
#include "ComputeSmallJ2PlasticityStress.h"

registerMooseObject("sharkApp", ComputeSmallJ2PlasticityStress);

InputParameters
ComputeSmallJ2PlasticityStress::validParams()
{
  InputParameters params = ComputeLagrangianObjectiveStress::validParams();
  params += SingleVariableReturnMappingSolution::validParams();
  params.addClassDescription("");
  params.addParam<MaterialPropertyName>(
      "elasticity_tensor", "elasticity_tensor", "The name of the elasticity tensor.");
  params.addParam<MaterialName>("hardening_model", "The hardening model.");
  params.addParam<MaterialPropertyName>("ge", 1.0, "The elasticity degradation function.");
  params.addParam<MaterialPropertyName>("gp", "The plasticity degradation function.");
  params.addParam<bool>("decompose_psip", false, "decompose the plastic energy for pff.");
  return params;
}

ComputeSmallJ2PlasticityStress::ComputeSmallJ2PlasticityStress(const InputParameters & parameters)
  : ComputeLagrangianObjectiveStress(parameters),
    GuaranteeConsumer(this),
    SingleVariableReturnMappingSolution(parameters),
    _elasticity_tensor_name(_base_name + getParam<MaterialPropertyName>("elasticity_tensor")),
    _elasticity_tensor(getMaterialProperty<RankFourTensor>(_elasticity_tensor_name)),
    _mechanical_strain_old(getMaterialPropertyOld<RankTwoTensor>(_base_name + "mechanical_strain")),
    _elastic_strain(declareProperty<RankTwoTensor>(_base_name + "elastic_strain")),
    _elastic_strain_old(getMaterialPropertyOld<RankTwoTensor>(_base_name + "elastic_strain")),
    _plastic_strain(declareProperty<RankTwoTensor>(_base_name + "plastic_strain")),
    _plastic_strain_old(getMaterialPropertyOld<RankTwoTensor>(_base_name + "plastic_strain")),
    _ep_name(_base_name + "effective_plastic_strain"),
    _ep(declareProperty<Real>(_ep_name)),
    _ep_old(getMaterialPropertyOldByName<Real>(_ep_name)),
    _Np(declareProperty<RankTwoTensor>(_base_name + "flow_direction")),
    _H(getMaterialPropertyByName<Real>(_base_name + "plastic_sigma")),
    _dH(getMaterialPropertyByName<Real>(_base_name + "plastic_dsigma")),
    _d2H(getMaterialPropertyByName<Real>(_base_name + "plastic_d2sigma")),
    _ge(getMaterialProperty<Real>("ge")),
    _gp(isParamValid("gp") ? getMaterialProperty<Real>("gp") : _ge),
    _psie_active(declareProperty<Real>(_base_name + "psie_active")),
    _decompose_psip(getParam<bool>("decompose_psip")),
    _strain_tr(declareProperty<Real>(_base_name + "strain_tr")),
    I2(RankTwoTensor::Identity()),
    Iijkl(I2.outerProduct(I2)),
    Iikjl(RankFourTensor::InitMethod::initIdentityFour),
    DEV(RankFourTensor::InitMethod::initIdentityDeviatoric)
// , SymDEV(0.5 * (DEV + DEV.transposeIj()))
{
}

void
ComputeSmallJ2PlasticityStress::initialSetup()
{
  // Enforce isotropic elastic tensor
  if (!hasGuaranteedMaterialProperty(_elasticity_tensor_name, Guarantee::ISOTROPIC))
    mooseError("ComputeSmallJ2PlasticityStress requires an isotropic elasticity tensor");

  _hardening_model = dynamic_cast<HardeningModel *>(&getMaterial("hardening_model"));
  _check_range = true;
}

void
ComputeSmallJ2PlasticityStress::initQpStatefulProperties()
{
  ComputeLagrangianObjectiveStress::initQpStatefulProperties();
  _ep[_qp] = 0;
  _psie_active[_qp] = 0.0;
}

void
ComputeSmallJ2PlasticityStress::computeQpSmallStress()
{
  usingTensorIndices(i, j, k, l, m);
  const Real G = ElasticityTensorTools::getIsotropicShearModulus(_elasticity_tensor[_qp]);
  const Real K = ElasticityTensorTools::getIsotropicBulkModulus(_elasticity_tensor[_qp]);

  _elastic_strain[_qp] = _mechanical_strain[_qp] - _plastic_strain_old[_qp];
  RankTwoTensor s = _ge[_qp] * 2 * G * _elastic_strain[_qp].deviatoric();
  Real s_norm = s.norm();
  _Np[_qp] = MooseUtils::absoluteFuzzyEqual(s_norm, 0) ? std::sqrt(1. / 2.) * I2
                                                       : std::sqrt(3. / 2.) * s / s_norm;
  Real s_eff = s.doubleContraction(_Np[_qp]);

  Real delta_ep = 0.;
  _hardening_model->setQp(_qp);

  if (computeResidual(s_eff, 0) > 0)
  {
    if (_fe_problem.currentlyComputingJacobian())
    {
      d_delta_ep_d_strain.zero();
      if (MooseUtils::absoluteFuzzyEqual(s_norm, 0))
      {
        dN_dstrain.zero();
      }
      else
      {
        auto dN_ds =
            std::sqrt(1.5) * (Iikjl / s_norm - s.times<i, j, k, l>(s) / std::pow(s_norm, 3));

        dN_dstrain = dN_ds * DEV * (2 * G * _ge[_qp]);
      }
    }
    returnMappingSolve(s_eff, delta_ep, _console);
    if (_fe_problem.currentlyComputingJacobian())
    {
      d_strain_new_d_strain =
          Iikjl - _Np[_qp].outerProduct(d_delta_ep_d_strain) - delta_ep * dN_dstrain;
    }
  }
  else
  {
    d_strain_new_d_strain = Iikjl;
  }

  // update strain
  _ep[_qp] = _ep_old[_qp] + delta_ep;
  RankTwoTensor elastic_correction = delta_ep * _Np[_qp];
  _elastic_strain[_qp] -= elastic_correction;
  _plastic_strain[_qp] = _plastic_strain_old[_qp] + elastic_correction;
  _strain_tr[_qp] = _elastic_strain[_qp].trace();

  Real H_pos = _strain_tr[_qp] > 0 ? 1. : 0.;

  RankTwoTensor stressA =
      H_pos * K * _strain_tr[_qp] * I2 + 2 * G * _elastic_strain[_qp].deviatoric();
  RankTwoTensor stressI = (1. - H_pos) * K * _strain_tr[_qp] * I2;

  _small_stress[_qp] = _ge[_qp] * stressA + stressI;

  _psie_active[_qp] = stressA.doubleContraction(_elastic_strain[_qp]);

  /**
   * update the plastic energy psip_active (psip) (in hardening model, not used here) and
   * psie_active. https://doi.org/10.1007/s00466-015-1225-3
   */
  _hardening_model->computeEnergy();
  if (_decompose_psip)
    _hardening_model->decomposePsip(H_pos);

  _hardening_model->computeEpdot();

  if (_fe_problem.currentlyComputingJacobian())
  {
    RankFourTensor dstress_delasticstrain =
        K * (_ge[_qp] * H_pos + 1 - H_pos) * Iijkl + 2 * _ge[_qp] * G * DEV;
    _small_jacobian[_qp] =
        dstress_delasticstrain * d_strain_new_d_strain * (Iikjl + I2.times<i, l, j, k>(I2)) / 2;
  }
}

Real
ComputeSmallJ2PlasticityStress::computeReferenceResidual(const Real & effective_trial_stress,
                                                         const Real & scalar)
{
  const Real G = ElasticityTensorTools::getIsotropicShearModulus(_elasticity_tensor[_qp]);
  return effective_trial_stress - 3 * _ge[_qp] * G * scalar;
}

Real
ComputeSmallJ2PlasticityStress::computeResidual(const Real & effective_trial_stress,
                                                const Real & scalar)
{

  const Real G = ElasticityTensorTools::getIsotropicShearModulus(_elasticity_tensor[_qp]);

  // Update the flow stress
  _ep[_qp] = _ep_old[_qp] + scalar;
  _hardening_model->computeHardeningStress();

  return effective_trial_stress - 3 * _ge[_qp] * G * scalar - _gp[_qp] * _H[_qp];
}

Real
ComputeSmallJ2PlasticityStress::computeDerivative(const Real & /*effective_trial_stress*/,
                                                  const Real & scalar)
{
  const Real G = ElasticityTensorTools::getIsotropicShearModulus(_elasticity_tensor[_qp]);
  _ep[_qp] = _ep_old[_qp] + scalar;
  _hardening_model->computeHardeningStressDerivative();
  return -3 * _ge[_qp] * G - _gp[_qp] * _dH[_qp];
}

void
ComputeSmallJ2PlasticityStress::preStep(const Real & scalar, const Real & R, const Real & J)
{
  if (!_fe_problem.currentlyComputingJacobian())
    return;
  const Real G = ElasticityTensorTools::getIsotropicShearModulus(_elasticity_tensor[_qp]);
  // Update the flow stress
  _ep[_qp] = _ep_old[_qp] + scalar;
  _hardening_model->computeHardeningStressDerivative();
  _hardening_model->computeHardeningStressDerivative2();

  dR_dstrain = 2 * _ge[_qp] * G * _Np[_qp].initialContraction(DEV) -
               (3 * _ge[_qp] * G + _gp[_qp] * _dH[_qp]) * d_delta_ep_d_strain;
  dJ_dstrain = -_gp[_qp] * _d2H[_qp] * d_delta_ep_d_strain;

  d_delta_ep_d_strain += -1 / J * dR_dstrain + R / J / J * dJ_dstrain;
}
// #endif