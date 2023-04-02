import trusspy as tp

# init model
M = tp.Model()

element_type   = 1    # truss
material_type  = 1    # linear-elastic

L1 = 1
L2 = 1
L3 = 1

E = 200*(10**9)

area  = 10**-4

F1 = 0
F2 = -(10**5)
F3 = 0

with M.Nodes as MN:
    MN.add_node( 1, coord=(0,    0,  0))
    MN.add_node( 2, coord=(0,    0,  L1))
    MN.add_node( 3, coord=(L2,   0,  0))
    MN.add_node( 4, coord=(L2,   0,  L1))
    MN.add_node( 5, coord=(L2+L3,0,  0))



with M.Elements as ME:
    ME.add_element( 1, conn=(1,3), gprop=[area] )
    ME.add_element( 2 ,conn=(1,4), gprop=[area] )
    ME.add_element( 3, conn=(2,3), gprop=[area] )
    ME.add_element( 4, conn=(2,4), gprop=[area] )
    ME.add_element( 5, conn=(3,4), gprop=[area] )
    ME.add_element( 6, conn=(3,5), gprop=[area] )
    ME.add_element( 7, conn=(4,5), gprop=[area] )

    ME.assign_etype(    'all',   element_type   )
    ME.assign_mtype(    'all',  material_type   )
    ME.assign_material( 'all', [E] )

with M.Boundaries as MB:
    MB.add_bound_U( 1, (0,0,0) )
    MB.add_bound_U( 2, (0,0,0) )
    MB.add_bound_U( 3, (1,0,1) )
    MB.add_bound_U( 4, (1,0,1) )
    MB.add_bound_U( 5, (1,0,1) )

with M.ExtForces as MF:
    MF.add_force( 5, (F1, F3, F2) )

# M.Settings.dlpf = 1
# M.Settings.du = 1

M.Settings.incs = 1

# M.Settings.stepcontrol = True
# M.Settings.maxfac = 1

# M.Settings.ftol = 10**8
# M.Settings.xtol = 1
# M.Settings.nfev = 1

# M.Settings.dxtol = 1

# build model and run job
M.build()
M.run()

# show results
M.plot_model(config=['deformed'],
             view='xz',
             contour='force',
             lim_scale=(-0.5,3.5,-2,2),
             force_scale=1/2000,
             inc=-1)

M.plot_history(nodes=[5,5], X='Displacement Z', Y='Force Z')

# M.plot_history(nodes=[5,5], X='Displacement X', Y='Displacement Z')

# M.plot_model(config=['undeformed'],
#                        view='xz', #'xy', 'yz', 'xz'
#                        contour='force',
#                     #    lim_scale=(-3,2,0,5,-1,4), #3d
#                        lim_scale=1.4, #plane-view
#                        force_scale=2.0, #5
#                        inc=0)

# M.plot_model(config=['deformed'],
#                        view='xz',
#                        contour='force',
#                        lim_scale=1.3,
#                        force_scale=500.0,
#                        inc=-1)

# M.plot_model(config=['deformed'],
#                        view='xz',
#                        contour='force',
#                     #    lim_scale=(-3,2,0,5,-2,3),
#                         lim_scale=1.3,
#                        force_scale=500,
#                        inc=-1)

M.plot_show()

# show results
M.plot_movie(config=['deformed'],
             view='xz',
             contour='force',
             lim_scale=(-0.5,3.5,-2,2),
             force_scale=5,
             cbar_limits=[-1,1])

print(M.Results.R[-1].U[-1])