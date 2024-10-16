#pragma once

#include "JohnsonCookNoTemp.h"

/**
 * Johnson-Cook hardening model.
 */
class JohnsonCook : public JohnsonCookNoTemp
{
public:
  static InputParameters validParams();

  JohnsonCook(const InputParameters & parameters);

protected:
  virtual Real computeAnalyticalHardeningStress(const Real & ep) override;
  virtual Real computeAnalyticalDerivative() override;
  virtual Real computeAnalyticalDerivative2() override;
  virtual Real temperatureFactor();

  const VariableValue & _T;
  const Real _T0;
  const Real _Tm;
  const Real _m;
};