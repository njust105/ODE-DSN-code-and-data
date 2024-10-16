#pragma once

#include "HardeningModel.h"

/**
 * Johnson-Cook hardening model (temperature independent).
 */
class JohnsonCookNoTemp : public HardeningModel
{
public:
  static InputParameters validParams();

  JohnsonCookNoTemp(const InputParameters & parameters);

  virtual Real initialGuess(const Real & effective_trial_stress) override;

protected:
  virtual Real computeAnalyticalHardeningStress(const Real & ep) override;
  virtual Real computeAnalyticalDerivative() override;
  virtual Real computeAnalyticalDerivative2() override;
  virtual Real rateFactor();

  virtual void computeHardeningStressRI() override;

  const Real _B;
  const Real _C;
  const Real _n;
  const Real _epdot0;
  const MaterialProperty<Real> & _epdot_old;
};