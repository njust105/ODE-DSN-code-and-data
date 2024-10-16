
#pragma once

#include "Material.h"

/**
 * HardeningModel base class.
 */
class HardeningModel : public Material
{
public:
  static InputParameters validParams();

  HardeningModel(const InputParameters & parameters);

  virtual void setQp(unsigned int qp) { _qp = qp; }

  virtual void computeEnergy();
  virtual void computeHardeningStress() final;
  virtual void computeHardeningStressDerivative() final;
  virtual void computeHardeningStressDerivative2() final;

  virtual Real initialGuess(const Real & effective_trial_stress) { return 0.0; }
  virtual void decomposePsip(const Real & heaviside_value) { _psip_active[_qp] *= heaviside_value; }
  virtual Real psip(const bool old = false) { return old ? _psip_old[_qp] : _psip[_qp]; }
  virtual void computeEpdot() { _epdot[_qp] = (_ep[_qp] - _ep_old[_qp]) / _dt; }

  void resetQpProperties() final {}
  void resetProperties() final {}
  void resetDerivativesAtQp();

protected:
  virtual void initQpStatefulProperties() override;

  virtual Real computeAnalyticalHardeningStress(const Real & ep) = 0;
  virtual Real computeAnalyticalDerivative();
  virtual Real computeAnalyticalDerivative2();

  // compute the rate-independent part of hardening stress.
  virtual void computeHardeningStressRI() { _sigma_RI[_qp] = _sigma[_qp]; }

  virtual Real thermalConjugate() { return 0.0; }

  const std::string _base_name;
  const bool _auto_derivation;
  const Real _eps;
  const MaterialProperty<Real> & _sigma_0;
  const Real _TQ;
  const MaterialProperty<Real> & _ep;
  const MaterialProperty<Real> & _ep_old;
  MaterialProperty<Real> & _epdot;
  MaterialProperty<Real> & _psip;
  const MaterialProperty<Real> & _psip_old;
  MaterialProperty<Real> & _psip_active;
  MaterialProperty<Real> & _heat_generation_rate;
  MaterialProperty<Real> & _sigma;
  const MaterialProperty<Real> & _sigma_old;
  MaterialProperty<Real> & _dsigma;
  MaterialProperty<Real> & _d2sigma;
  MaterialProperty<Real> & _sigma_RI;
  const MaterialProperty<Real> & _sigma_RI_old;
};