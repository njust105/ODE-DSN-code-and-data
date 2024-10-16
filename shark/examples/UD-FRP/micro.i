end_time = 6.66e-6
dtmax = '${fparse end_time / 50}'
dt = '${fparse end_time / 50}'

matrix_E = 2.956e3
matrix_nu = 0.35
matrix_A = 81.3
matrix_C = 546.15
matrix_P = 2.22

fiber_E1 = 294e3
fiber_E2 = 15e3
fiber_E3 = 15e3
fiber_nu12 = 0.2
fiber_nu13 = 0.2
fiber_nu23 = 0.02
fiber_G12 = 19e3
fiber_G13 = 19e3
fiber_G23 = 9e3

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
    file = randomFiber-3d.msh
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
  [e110]
    type = PiecewiseLinear
    format = columns
    xy_in_file_only = false
    data_file = ${csvfile}
    x_title = 'time'
    y_title = 'x0'
    extrap = true
  []
  [e11]
    type = ParsedFunction
    expression = ' e110'
    symbol_names = 'e110'
    symbol_values = 'e110'
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
    type = CowperSymonds
    sigma_0 = ${matrix_A}
    C = ${matrix_C}
    P = ${matrix_P}
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
    optimal_iterations = 5
    linear_iteration_ratio = 25
    dt = ${dt}
  []
  dtmin = 1e-8
  dtmax = ${dtmax}
  end_time = ${end_time}
[]

[Outputs]
  exodus = true
  color = true
  csv = true
[]
