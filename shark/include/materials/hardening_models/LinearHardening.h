#pragma once

#include "HardeningModel.h"

/**
 * Linear hardening model.
 */
class LinearHardening : public HardeningModel
{
public:
  static InputParameters validParams();

  LinearHardening(const InputParameters & parameters);

protected:
  virtual Real computeAnalyticalHardeningStress(const Real & ep) override;
  virtual Real computeAnalyticalDerivative() override;
  virtual Real computeAnalyticalDerivative2() override { return 0.0; }

  const MaterialProperty<Real> & _H;
};