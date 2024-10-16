// #ifdef DONOTCOMPLIE
#pragma once

#include "ComputeLagrangianObjectiveStress.h"
#include "GuaranteeConsumer.h"
#include "ElasticityTensorTools.h"
#include "SingleVariableReturnMappingSolution.h"
#include "Function.h"
#include "HardeningModel.h"

/**
 *
 */
class ComputeSmallJ2PlasticityStress : public ComputeLagrangianObjectiveStress,
                                       public GuaranteeConsumer,
                                       public SingleVariableReturnMappingSolution
{
public:
  static InputParameters validParams();

  ComputeSmallJ2PlasticityStress(const InputParameters & parameters);

  virtual void initialSetup() override;

protected:
  virtual void initQpStatefulProperties() override;

  virtual void computeQpSmallStress() override;

  /// @{ The return mapping residual and derivative
  virtual Real computeReferenceResidual(const Real & effective_trial_stress,
                                        const Real & scalar) override;
  virtual Real computeResidual(const Real & effective_trial_stress, const Real & scalar) override;
  virtual Real computeDerivative(const Real & effective_trial_stress, const Real & scalar) override;
  virtual void
  preStep(const Real & scalar_old, const Real & residual, const Real & jacobian) override;
  virtual Real initialGuess(const Real & effective_trial_stress) override
  {
    return _hardening_model->initialGuess(effective_trial_stress);
  }
  virtual Real minimumPermissibleValue(const Real &) const override { return 0; }
  /// @}

  const MaterialPropertyName _elasticity_tensor_name;
  const MaterialProperty<RankFourTensor> & _elasticity_tensor;

  const MaterialProperty<RankTwoTensor> & _mechanical_strain_old;
  MaterialProperty<RankTwoTensor> & _elastic_strain;
  const MaterialProperty<RankTwoTensor> & _elastic_strain_old;
  MaterialProperty<RankTwoTensor> & _plastic_strain;
  const MaterialProperty<RankTwoTensor> & _plastic_strain_old;

  const std::string _ep_name;
  MaterialProperty<Real> & _ep;
  const MaterialProperty<Real> & _ep_old;
  MaterialProperty<RankTwoTensor> & _Np;
  const MaterialProperty<Real> & _H;
  const MaterialProperty<Real> & _dH;
  const MaterialProperty<Real> & _d2H;

  const MaterialProperty<Real> & _ge;
  const MaterialProperty<Real> & _gp;

  MaterialProperty<Real> & _psie_active;

  const bool _decompose_psip;

  MaterialProperty<Real> & _strain_tr;

  HardeningModel * _hardening_model;

private:
  const RankTwoTensor I2;
  const RankFourTensor Iijkl;
  const RankFourTensor Iikjl;
  const RankFourTensor DEV;
  // const RankFourTensor SymDEV;

  // $$\frac{\partial \Delta p}{\partial \varepsilon^{trial}}$$
  RankTwoTensor d_delta_ep_d_strain;
  RankFourTensor d_strain_new_d_strain;
  RankTwoTensor dR_dstrain;
  RankTwoTensor dJ_dstrain;

  RankFourTensor dN_dstrain;
};

// #endif