#pragma once

#include "HardeningModel.h"

/**
 * Johnson-Cook hardening model (temperature independent).
 */
class CowperSymonds : public HardeningModel
{
public:
  static InputParameters validParams();

  CowperSymonds(const InputParameters & parameters);

  virtual Real initialGuess(const Real & effective_trial_stress) override;

protected:
  virtual Real computeAnalyticalHardeningStress(const Real & ep) override;

  const Real _C;
  const Real _P;
};