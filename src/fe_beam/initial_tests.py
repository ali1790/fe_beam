import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from fe_beam.core.mesh import ElementConnectivity, Mesh, Node
from fe_beam.elements.timoshenko_beam import SectionConstitutive, TimoshenkoBeamElement
from fe_beam.core.dof import DofManager
from fe_beam.core.assembly import assemble_global_matrices
from fe_beam.core.boundary_conditions import DirichletBC, NeumannBC, build_load_vector, apply_dirichlet_static, apply_dirichlet_harmonic

def create_ansys_matrices_circle_euler(r):
    A = np.pi * r**2
    E_steel = 200E9 # Pa
    G_steel = 79.3
    rho_steel = 8E3 #kg/m^3
    I =  0.25 * np.pi * r**4.
    J = 0.5 * np.pi * r**4

    cbmx = np.zeros((6,6))
    cbmx[0,0] = E_steel * A
    cbmx[1,1] = E_steel * I
    cbmx[2,2] = E_steel * I
    cbmx[3,3] = G_steel * J

    cbmd = np.zeros((6,6))


    cbmd[0, 0] = rho_steel * A
    cbmd[1, 1] = rho_steel * A
    cbmd[2, 2] = rho_steel * A
    cbmd[3, 3] = 0.5 * rho_steel  * J
    cbmd[4, 4] = 0.5 * rho_steel  * J
    cbmd[5, 5] = 0.5 * rho_steel  * J
    return cbmx, cbmd

def run_test():
    # Create mesh
    l_beam = 10. #m
    n_nodes = 10
    r_beam = 0.1 # m
    mesh = create_mesh(l_beam, n_nodes)

    # Assign material properties to elements
    #cbmx, cbmd = create_ansys_matrices_circle_euler(r_beam)
    cbmx = np.diag([1e7, 1e5, 1e5, 1e5, 5e4, 5e4])
    cbmd = np.diag([10.0, 1.0, 1.0, 1.0, 0.5, 0.5])
    section = SectionConstitutive(
                S=cbmx,
                C=cbmd,
                # Falls ANSYS-Reihenfolge abweicht: hier anpassen
                order=("ex", "kx", "ky", "kz", "gy", "gz"),
                )
    beam_elements = []
    for element in mesh.get_all_elements():
        beam_elements.append( 
            TimoshenkoBeamElement(
                element.element_id,
                element.node_ids,
                mesh.get_element_length(element.element_id),
                section
            )
        )
    # Create global system matrices
    dof_manager = DofManager()
    dof_manager.enumerate_dofs(beam_elements)
    ndof = dof_manager.number_of_dofs()
    system = assemble_global_matrices(mesh, beam_elements, dof_manager)

    dirichlet_bcs = [
        DirichletBC(0, "u", 0.0),
        DirichletBC(0, "v", 0.0),
        DirichletBC(0, "w", 0.0),
        DirichletBC(0, "phix", 0.0),
        DirichletBC(0, "phiy", 0.0),
        DirichletBC(0, "phiz", 0.0),
    ]
    neumann_bcs = [
        NeumannBC(n_nodes-1, "u", -1000.0)  # downward force at free end
    ]

    f_static = build_load_vector(
        dof_manager=dof_manager,
        ndof=ndof,
        neumann_bcs=neumann_bcs,
        dtype=float
    )
    reduced_system = apply_dirichlet_static(
        K=system.K,
        f=f_static,
        dof_manager=dof_manager,
        dirichlet_bcs=dirichlet_bcs
    )
    u_free = spsolve(reduced_system.A, reduced_system.f)
    u_full = reduced_system.reconstruct_full_solution(u_free)

    sol = np.zeros((n_nodes, 9))
    for i in range(n_nodes):
        sol[i, :3] = mesh.nodes[i].coordinates()

        u = u_full[dof_manager.get_dof_index(i, "u")]
        v = u_full[dof_manager.get_dof_index(i, "v")]
        w = u_full[dof_manager.get_dof_index(i, "w")]

        phix = u_full[dof_manager.get_dof_index(i, "phix")]
        phiy = u_full[dof_manager.get_dof_index(i, "phiy")]
        phiz = u_full[dof_manager.get_dof_index(i, "phiz")]

        sol[i, 3] = u
        sol[i, 4] = v
        sol[i, 5] = w

        sol[i, 6] = phix
        sol[i, 7] = phiy
        sol[i, 8] = phiz
    plot_solution(sol)

def plot_solution(solution):
    fig, axs = plt.subplots(ncols=2)
    axs[0].plot(solution[:, 2], solution[:, 3], label='Ux')
    axs[0].plot(solution[:, 2], solution[:, 4], label='Uy')
    axs[0].plot(solution[:, 2], solution[:, 5], label='Uz')
    axs[0].legend()
    
    axs[1].plot(solution[:, 2], solution[:, 6], label=r'$\phi_x$')
    axs[1].plot(solution[:, 2], solution[:, 7], label=r'$\phi_y$')
    axs[1].plot(solution[:, 2], solution[:, 8], label=r'$\phi_z$')
    axs[1].legend()
    plt.show()

def create_mesh(l_beam, n_nodes):
    mesh = Mesh()
    # Create nodes for beam pointing in z direction
    z = np.linspace(0, l_beam, endpoint=True, num=n_nodes )
    for i in range(n_nodes):
        mesh.add_node( Node(i, z[i], 0., 0.) )

    # Create elements 
    for i in range(n_nodes - 1):
        mesh.add_element( ElementConnectivity( i, [i, i + 1] ) )
    
    #mesh.show()
    return mesh

if __name__=='__main__':
    run_test()
    