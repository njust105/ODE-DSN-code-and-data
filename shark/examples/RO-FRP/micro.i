end_time = 1.3e-4
dtmax = '${fparse end_time / 50}'
dtmin = '${fparse end_time / 400}'
dt = '${fparse end_time / 50}'
# http://dx.doi.org/10.1016/j.compstruct.2014.12.061
matrix_E = 3.6e3
matrix_nu = 0.4
matrix_A = 132
matrix_B = 10
matrix_C = 0.034
matrix_n = 1.2
matrix_epdot0 = 1e-3

# http://dx.doi.org/10.18419/opus-3848
fiber_E1 = 74e3
fiber_E2 = 74e3
fiber_E3 = 74e3
fiber_nu12 = 0.2
fiber_nu13 = 0.2
fiber_nu23 = 0.2
fiber_G12 = 74e3
fiber_G13 = 74e3
fiber_G23 = 30.8e3

fiber_nu21 = '${fparse fiber_nu12 * fiber_E2 / fiber_E1}'
fiber_nu31 = '${fparse fiber_nu13 * fiber_E3 / fiber_E1}'
fiber_nu32 = '${fparse fiber_nu23 * fiber_E3 / fiber_E2}'

csvfile = 'test.csv'

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
  large_kinematics = false
  # macro_gradient = hvar
  # homogenization_constraint = homogenization
  dim = 3
[]

[Mesh]
  [base]
    type = FileMeshGenerator
    file = 28-1727201711-6.4716160636325775.msh
  []
  [fix]
    type = BoundingBoxNodeSetGenerator
    input = base
    bottom_left = '-0.001 -0.001 -0.001'
    top_right = '0.001  0.001 0.001'
    new_boundary = fix
  []
  [fix_x]
    type = BoundingBoxNodeSetGenerator
    input = fix
    bottom_left = '0.1999 -0.001 -0.001'
    top_right = '0.2001 0.001 0.001'
    new_boundary = fix_x
  []
  [fix_y]
    type = BoundingBoxNodeSetGenerator
    input = fix_x
    bottom_left = '-0.001 0.1999 -0.001'
    top_right = '0.001 0.2001 0.001'
    new_boundary = fix_y
  []
[]

[Variables]
  [disp_x]
  []
  [disp_y]
  []
  [disp_z]
  []
  [hvar]
    family = SCALAR
    order = SIXTH
  []
[]

[Functions]
  [e11]
    type = PiecewiseLinear
    format = columns
    xy_in_file_only = false
    data_file = ${csvfile}
    x_title = 'time'
    y_title = 'x0'
    extrap = true
  []
  [e22]
    type = PiecewiseLinear
    format = columns
    xy_in_file_only = false
    data_file = ${csvfile}
    x_title = 'time'
    y_title = 'x1'
    extrap = true
  []
  [e33]
    type = PiecewiseLinear
    format = columns
    xy_in_file_only = false
    data_file = ${csvfile}
    x_title = 'time'
    y_title = 'x2'
    extrap = true
  []
  [e23]
    type = PiecewiseLinear
    format = columns
    xy_in_file_only = false
    data_file = ${csvfile}
    x_title = 'time'
    y_title = 'x3'
    extrap = true
  []
  [e13]
    type = PiecewiseLinear
    format = columns
    xy_in_file_only = false
    data_file = ${csvfile}
    x_title = 'time'
    y_title = 'x4'
    extrap = true
  []
  [e12]
    type = PiecewiseLinear
    format = columns
    xy_in_file_only = false
    data_file = ${csvfile}
    x_title = 'time'
    y_title = 'x5'
    extrap = true
  []
[]

[UserObjects]
  [homogenization]
    type = HomogenizationConstraint
    constraint_types = 'strain none none strain strain none strain strain strain'
    targets = 'e11 e12 e22 e13 e23 e33'
    execute_on = 'INITIAL LINEAR NONLINEAR'
  []
[]

[Kernels]
  [sdx]
    type = HomogenizedTotalLagrangianStressDivergence
    variable = disp_x
    component = 0
    homogenization_constraint = homogenization
    macro_gradient = hvar
  []
  [sdy]
    type = HomogenizedTotalLagrangianStressDivergence
    variable = disp_y
    component = 1
    homogenization_constraint = homogenization
    macro_gradient = hvar
  []
  [sdz]
    type = HomogenizedTotalLagrangianStressDivergence
    variable = disp_z
    component = 2
    homogenization_constraint = homogenization
    macro_gradient = hvar
  []
[]

[ScalarKernels]
  [enforce]
    type = HomogenizationConstraintScalarKernel
    variable = hvar
    homogenization_constraint = homogenization
  []
[]

[BCs]
  [Periodic]
    [x]
      variable = disp_x
      auto_direction = 'x y z'
    []
    [y]
      variable = disp_y
      auto_direction = 'x y z'
    []
    [z]
      variable = disp_z
      auto_direction = 'x y z'
    []
  []
  [fix1_x]
    type = DirichletBC
    boundary = fix
    variable = disp_x
    value = 0
  []
  [fix1_y]
    type = DirichletBC
    boundary = fix
    variable = disp_y
    value = 0
  []
  [fix1_z]
    type = DirichletBC
    boundary = fix
    variable = disp_z
    value = 0
  []
  [fixr_z]
    type = DirichletBC
    boundary = "fix_x"
    variable = disp_y
    value = 0
  []
  [fixr_y]
    type = DirichletBC
    boundary = "fix_x"
    variable = disp_z
    value = 0
  []
  [fixr_x]
    type = DirichletBC
    boundary = "fix_y"
    variable = disp_z
    value = 0
  []
[]

[Materials]
  [compute_strain]
    type = ComputeLagrangianStrain
    homogenization_gradient_names = 'homogenization_gradient'
  []
  [compute_homogenization_gradient]
    type = ComputeHomogenizedLagrangianStrain
    homogenization_constraint = homogenization
    macro_gradient = hvar
  []

  [elastic_tensor_matrix]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = ${matrix_E}
    poissons_ratio = ${matrix_nu}
    block = matrix
  []

  [elastic_tensor_fiber]
    type = ComputeElasticityTensor
    fill_method = orthotropic
    block = fiber
    C_ijkl = '${fiber_E1} ${fiber_E2} ${fiber_E3} 
              ${fiber_G12} ${fiber_G23} ${fiber_G13} 
              ${fiber_nu21} ${fiber_nu31} ${fiber_nu32} 
              ${fiber_nu12} ${fiber_nu13} ${fiber_nu23}'
  []

  [hardening_model_matrix]
    type = JohnsonCookNoTemp
    sigma_0 = ${matrix_A}
    B = ${matrix_B}
    C = ${matrix_C}
    n = ${matrix_n}
    epdot0 = ${matrix_epdot0}
    auto_derivation = true
    block = matrix
  []

  [compute_stress_matrix]
    type = ComputeSmallJ2PlasticityStress
    hardening_model = hardening_model_matrix
    block = matrix
  []
  [compute_stress_fiber]
    type = ComputeLagrangianLinearElasticStress
    block = fiber
  []

[]

[Postprocessors]
  [AverageSmallStressStrain]
    [all]
      dim = 3
      strain_pp_base_name = E
      stress_pp_base_name = S
    []
  []
[]

[Executioner]
  type = Transient
  solve_type = NEWTON
  line_search = none
  petsc_options_iname = '-pc_type -snes_type -pc_factor_mat_solver_package'
  petsc_options_value = 'lu vinewtonrsls superlu_dist'

  nl_max_its = 50
  nl_rel_tol = 1e-08
  nl_abs_tol = 1e-10
  [TimeStepper]
    type = IterationAdaptiveDT
    optimal_iterations = 10
    linear_iteration_ratio = 25
    dt = ${dt}
  []
  dtmin = ${dtmin}
  dtmax = ${dtmax}
  end_time = ${end_time}
[]

[Outputs]
  exodus = true
  color = true
  csv = true
[]
