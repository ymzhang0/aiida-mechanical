from re import A
import numpy as np
import numpy.linalg as la
from pymatgen.core.operations import SymmOp

_rotation_matrices_cartesian = {
    'identity': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    'rot.60 deg, x-axis': [[1, 0, 0], [0, 0.5, -np.sqrt(3)/2], [0, np.sqrt(3)/2, 0.5]],
    'rot.60 deg, y-axis': [[0.5, 0, np.sqrt(3)/2], [0, 1, 0], [-np.sqrt(3)/2, 0, 0.5]],
    'rot.60 deg, z-axis': [[0.5, -np.sqrt(3)/2, 0], [np.sqrt(3)/2, 0.5, 0], [0, 0, 1]],
    'rot.90 deg, x-axis': [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    'rot.90 deg, y-axis': [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    'rot.90 deg, z-axis': [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    'rot.120 deg, x-axis': [[1, 0, 0], [0, -0.5, -np.sqrt(3)/2], [0, np.sqrt(3)/2, -0.5]],
    'rot.120 deg, y-axis': [[-0.5, 0, np.sqrt(3)/2], [0, 1, 0], [-0.5, 0, -np.sqrt(3)/2]],
    'rot.120 deg, z-axis': [[-0.5, -np.sqrt(3)/2, 0], [np.sqrt(3)/2, -0.5, 0], [0, 0, 1]],
    'rot.180 deg, x-axis': [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
    'rot.180 deg, y-axis': [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
    'rot.180 deg, z-axis': [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
    'rot.-60 deg, x-axis': [[1, 0, 0], [0, 0.5, np.sqrt(3)/2], [0, -np.sqrt(3)/2, 0.5]],
    'rot.-60 deg, y-axis': [[0.5, 0, -np.sqrt(3)/2], [0, 1, 0], [np.sqrt(3)/2, 0, 0.5]],
    'rot.-60 deg, z-axis': [[0.5, np.sqrt(3)/2, 0], [-np.sqrt(3)/2, 0.5, 0], [0, 0, 1]],
    }

ibrav_bravais_lattice_map_qe = {
    1: 'CUB',
    2: 'FCC',
    3: 'BCC',
    4: 'HEX',
    5: 'RHL',
    6: 'TET',
    7: 'BCT',
    8: 'ORC',
    9: 'ORCC',
    91: 'ORCC-A',
    10: 'ORCF',
    11: 'ORCI',
    12: 'MCL-A',
    -12: 'MCL-B',
    13: 'MCLC-C',
    -13: 'MCLC-B',
    14: 'TRI',
}

ibrav_bravais_lattice_map_ase = {
    'CUB': 1,
    'FCC': 2,
    'BCC': 3,
    'HEX': 4,
    'RHL': 5,
    'TET': 6,
    'BCT': 7,
    'ORC': 8,
    'ORCC': 9,
    'ORCF': 10,
    'ORCI': 11,
    'MCL': 12,
    'MCLC': 13,
    'TRI': 14,
}


def is_identity(left_matrix, right_matrix, atol=1e-6):
    return np.allclose(left_matrix, right_matrix, atol=atol)

def get_ibrav_ase(
    structure,
    eps=1e-6,
    ):
    """Get the ibrav of the structure."""

    ase_atoms = structure.get_ase()
    cell = ase_atoms.cell

    bravais_lattice = cell.get_bravais_lattice(eps=eps)
    lattice_name = bravais_lattice.name

    if lattice_name in ibrav_bravais_lattice_map_ase:
        ibrav = ibrav_bravais_lattice_map_ase[lattice_name]
    else:
        ibrav = 0

    return ibrav

def base_transformation(
    V_initial,
    V_final,
    eps=1e-6
    ) -> tuple:
    """
    decompose the transformation from V_initial to V_final into a physical rotation (R) and a basis change (T).

    Args:
        V_initial (3x3 np.ndarray): the initial cell matrix (row vectors).
        V_final (3x3 np.ndarray): the final cell matrix (row vectors).

    Returns:
        a dictionary containing the rotation matrix 'R' and the basis change matrix 'T'.
    """
    V_initial = np.array(V_initial)
    V_final = np.array(V_final)

    if not (V_initial.shape == (3, 3) and V_final.shape == (3, 3)):
        raise ValueError("The cell matrix must be 3x3.")

    # --- First check if there is a pure integer basis change ---
    T_candidate = V_final @ np.linalg.inv(V_initial)
    T_rounded = np.round(T_candidate)

    # Check if T_candidate is close to an integer matrix
    is_integer_transform = np.allclose(T_candidate, T_rounded, atol=eps)

    if is_integer_transform:
        # If so, then this is a pure basis change, no physical rotation
        return T_rounded.astype(int)

    else:
        raise ValueError("The cell matrix is not a pure basis change.")

def get_cell_qe_convention(
    cell,
    eps=1e-6,
    ):
    """
    Get the basis vectors and cell parameters in the Quantum ESPRESSO convention.
    This method will check all the possible cases where the given cell is equivalent to cell in Quantum ESPRESSO convention modulo a rotation or linear transformation.

    Args:
        cell (list): the cell parameters in the A, B, C, cosAB, cosAC, cosBC form.
        eps (float): the tolerance for the cell parameters.

    Returns:
        a dictionary containing the cell parameters in the Quantum ESPRESSO convention.
    """

    # Here we consider a general case in conditioning where a cell is equivalent to some ibrav upon a rotation.
    # So after determining the ibrav, we should rotate the atom positions along with the cell definition.

    # cubic cell if three vectors have the same length and are mutually orthogonal
    v1, v2, v3 = [np.array(v) for v in cell]

    a, b, c, cosab, cosac, cosbc = get_cellpar(cell)

    parameters = {
        'ibrav': 0
        }
    cell_qe = None

    # --- CUBIC CELL --- #
    # cubic cell if three vectors have the same length and are mutually orthogonal
    # Note: the cell is allowed to have arbitrary rotation.
    if (
        (max(a, b, c) - min(a, b, c)) <= eps
        and
        max(abs(cosab), abs(cosac), abs(cosbc)) <= eps
        ):
        parameters['ibrav'] = 1
        parameters['a'] = a
        cell_qe = a * np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

    # --- FCC CELL --- #
    # FCC cell if three vectors have the same length and there angles are either 60° or 120°
    # Note: there are 12 nearest neighbours to the origin. All of them are equivalent to the origin.
    # Note: the cell is allowed to have arbitrary rotation.
    elif (
        (max(a, b, c) - min(a, b, c)) <= eps
        and
        max(abs(abs(cosab) - 1/2), abs(abs(cosac) - 1/2), abs(abs(cosbc) - 1/2)) <= eps
    ):
        a = a*2**0.5
        parameters['ibrav'] = 2
        parameters['a'] = a
        cell_qe = a / 2 * np.array([
            [-1, 0, 1],
            [ 0, 1, 1],
            [-1, 1, 0],
        ])

    # --- BCC CELL --- #
    # BCC cell if three vectors have the same length and there angles are either arccos(1/9) or arccos(-1/9)
    # Note: there are 8 nearest neighbours to the origin. All of them are equivalent to the origin.
    # Note: the cell is allowed to have arbitrary rotation.
    elif (
        (max(a, b, c) - min(a, b, c)) <= eps
        and
        max(abs(abs(cosab) - 1/3), abs(abs(cosac) - 1/3), abs(abs(cosbc) - 1/3)) <= eps
    ):
        a = a*2/3**0.5
        parameters['ibrav'] = 3
        parameters['a'] = a
        cell_qe = a / 2* np.array([
            [ 1,  1, 1],
            [-1,  1, 1],
            [-1, -1, 1],
        ])

    # --- HEXAGONAL CELL --- #
    # hexagonal cell if <v1, v2> have the same length and angle 60° or 120° and v3 is perpendicular to v1 and v2
    # Note: there are 6 nearest neighbours to the origin. All of them are equivalent to the origin.
    # Note: the cell is allowed to have arbitrary rotation.
    elif (
        abs(a - b) <= eps
        and
        max(abs(abs(cosab) - 1/2), abs(cosac), abs(cosbc)) <= eps
    ):
        parameters['ibrav'] = 4
        parameters['a'] = a
        parameters['c'] = c
        cell_qe = a * np.array([
            [1, 0, 0],
            [-1/2, np.sqrt(3)/2, 0],
            [0, 0, c/a],
        ])

    # --- Rhombohedral CELL --- #
    # rhombohedral cell if three vectors have the same length and there angles are the same.
    # Note: unlike Quantum ESPRESSO, thermo_pw.x only accept definition of cosab although in rhombohedral cell although cosab == cosac == cosbc.
    # Note: there are 6 nearest neighbours to the origin. All of them are equivalent to the origin.
    # Note: the cell is allowed to have arbitrary rotation.
    elif (
        (max(a, b, c) - min(a, b, c)) <= eps
        and
        max(cosab, cosac, cosbc) - min(cosab, cosac, cosbc) < eps
        ):
        parameters['ibrav'] = 5
        parameters['a'] = a
        parameters['cosab'] = cosab

        tx = np.sqrt((1-cosab)/2)
        ty = np.sqrt((1-cosab)/6)
        tz = np.sqrt((1+2*cosab)/3)
        cell_qe = a * np.array([
            [tx, -ty, tz],
            [0, 2*ty, tz],
            [-tx, -ty, tz],
        ])

    # --- TETRAGONAL CELL --- #
    # tetragonal cell if two vectors have the same length and are perpendicular to each other. The third vector is perpendicular to the plane of the first two vectors.
    # Note: there are 4 nearest neighbours to the origin. All of them are equivalent to the origin.
    # Note: the cell is allowed to have arbitrary rotation.
    elif (
        abs(a - b) <= eps
        and
        max(cosab, cosac, cosbc) - min(cosab, cosac, cosbc) < eps
        ):
        parameters['ibrav'] = 6
        parameters['a'] = a
        parameters['c'] = c
        cell_qe = a * np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, c/a],
        ])

    # --- BODY-CENTRED TETRAGONAL CELL --- #
    # body-centred tetragonal cell if three vectors have the following pattern:
    # v1=(a/2)(1,-1,c/a),  v2=(a/2)(1,1,c/a),  v3=(a/2)(-1,-1,c/a)
    # Two degrees of freedom: a and c are determined by the six cell parameters and the following constraints:
    # 1. a == b == c --> 4 degrees of freedom
    # 2. |cos(ab)| + |cos(bc)| + |cos(ac)| == 1 --> 3 degrees of freedom
    # 3. either cos(ab) == cos(bc) or cos(ab) == cos(ac) or cos(bc) == cos(ac) --> 2 degrees of freedom
    elif (
        max(a, b, c) - min(a, b, c) < eps
        and
        (
            abs(cosab - cosbc) < eps and abs(abs(cosab) + abs(cosbc) - cosac - 1) < eps
            or
            abs(cosab - cosac) < eps and abs(abs(cosab) + abs(cosac) - cosbc - 1) < eps
            or
            abs(cosbc - cosac) < eps and abs(abs(cosbc) + abs(cosac) - cosab - 1) < eps
        )
    ):
        # We should switch the special axis to v1
        if abs(cosab - cosac) < eps:
            a = ((a**2 - a*b*abs(cosab))*2)**0.5
            c = (a*b*abs(cosab)*4)**0.5
        elif abs(cosab - cosbc) < eps:
            a = ((a**2 - a*b*abs(cosab))*2)**0.5
            c = (a*b*abs(cosab)*4)**0.5
        elif abs(cosbc - cosac) < eps:
            a = ((a**2 - b*c*abs(cosbc))*2)**0.5
            c = (b*c*abs(cosbc)*4)**0.5

        parameters['ibrav'] = 7
        parameters['a'] = a
        parameters['c'] = c
        cell_qe = a / 2 * np.array([
            [1, -1, c/a],
            [1, 1, c/a],
            [-1, -1, c/a],
        ])

    # --- ORTHORHOMBIC CELL --- #
    # Orthorhombic cell if three vectors are mutually orthogonal.
    # Note: there are 4 nearest neighbours to the origin. All of them are equivalent to the origin.
    # Note: the cell is allowed to have arbitrary rotation.
    elif (
        max(cosab, cosac, cosbc) - min(cosab, cosac, cosbc) < eps
    ):
        parameters['ibrav'] = 8
        parameters['a'] = a
        parameters['b'] = b
        parameters['c'] = c
        cell_qe = np.array([
            [a, 0, 0],
            [0, b, 0],
            [0, 0, c],
        ])

    # --- BASE-CENTRED ORTHORHOMBIC CELL --- #
    # base-centred orthorhombic cell if three vectors have the following pattern:
    # v1 = (a/2, b/2,0),  v2 = (-a/2,b/2,0),  v3 = (0,0,c)
    # So v1 and v2 should have equal length and v3 should be perpendicular to v1 - v2.
    # Note: the cell is allowed to have arbitrary rotation.
    elif (
        max(abs(a - b), abs(cosac), abs(cosbc)) < eps
    ):
        x = (2*(a**2 - a**2*cosab))**0.5
        y = (2*(a**2 + a**2*cosab))**0.5
        parameters['ibrav'] = 9
        parameters['a'] = x
        parameters['b'] = y
        parameters['c'] = c
        cell_qe = np.array([
            [x/2, y/2, 0],
            [-x/2, y/2, 0],
            [0, 0, c],
        ])
    elif (
        abs(a**2-a*b*cosab-a*c*cosac) < eps
        and
        abs(b**2-a*b*cosab-b*c*cosbc) < eps
        and
        abs(c**2-a*c*cosac-b*c*cosbc) < eps
    ):
        parameters['ibrav'] = 10
        parameters['a'] = (4*a*b*cosab)**0.5
        parameters['b'] = (4*b*c*cosac)**0.5
        parameters['c'] = (4*a*c*cosbc)**0.5
        cell_qe = 1/2 * np.array([
            [a, 0, c],
            [a, b, 0],
            [0, b, c],
        ])
    elif (
        max(a, b, c) - min(a, b, c) < eps
        and
        abs(cosab+cosbc+cosac+1) < eps
    ):
        parameters['ibrav'] = 11
        a = a*((1+cosbc)*2)**0.5
        b = b*((1+cosac)*2)**0.5
        c = c*((1+cosab)*2)**0.5
        parameters['a'] = a
        parameters['b'] = b
        parameters['c'] = c
        cell_qe = 0.5* np.array([
            [ a,  b,  c],
            [-a,  b,  c],
            [-a, -b,  c],
        ])

    # --- MONOCLINIC CELL --- #
    # Monoclinic cell if one axis is perpendicular to the plane of the other two axes.
    # Note: the cell is allowed to have arbitrary rotation.
    elif (
        max(abs(cosac), abs(cosbc)) < eps
        ):
        parameters['ibrav'] = 12
        parameters['a'] = a
        parameters['b'] = b
        parameters['c'] = c
        parameters['cosab'] = cosab
        cell_qe = np.array([
            [a, 0, 0],
            [b*cosab, b*np.sqrt(1-cosab**2), 0],
            [0, 0, c],
        ])
    elif (
        max(abs(cosab), abs(cosbc)) < eps
        ):
        parameters['ibrav'] = -12
        parameters['a'] = a
        parameters['b'] = b
        parameters['c'] = c
        parameters['cosac'] = cosac
        cell_qe = np.array([
            [a, 0, 0],
            [b*cosab, b*np.sqrt(1-cosab**2), 0],
            [0, 0, c],
        ])

    # --- BASE-CENTRED MONOCLINIC CELL --- #
    # base-centred monoclinic cesll should follow the following pattern:
    # v1 = (  a/2,         0,          -c/2),
    # v2 = (b*cos(gamma), b*sin(gamma), 0  ),
    # v3 = (  a/2,         0,           c/2),
    # So v1 and v3 should have equal length and v2 should be perpendicular to v1 - v3.
    # We allow rotation in {v1, v3} plane.
    elif (
        abs(a - c) < eps
        ):
        if abs(cosab+cosbc) < eps:
            print('Found a base-centred monoclinic cell, special axis: v2')
            parameters['ibrav'] = 13
            _c = a*((1+cosac)*2)**0.5
            _b = 2*b
            _a = a*((1-cosac)*2)**0.5
            parameters['a'] = _a
            parameters['b'] = _b
            parameters['c'] = _c
            parameters['cosab'] = cosab
            cell_qe = 1/2 * np.array([
                [_a, 0, -_c],
                [_b*cosab, _b*np.sqrt(1-cosab**2), 0],
                [_a, 0, _c],
            ])
        if abs(cosab-cosbc) < eps:
            print('Found a base-centred monoclinic cell, special axis: v2')
            parameters['ibrav'] = 13
            _a = a*((1+cosac)*2)**0.5
            _b = 2*b
            _c = a*((1-cosac)*2)**0.5
            parameters['a'] = _a
            parameters['b'] = _b
            parameters['c'] = _c
            parameters['cosab'] = cosab
            cell_qe = 1/2 * np.array([
                [_a, 0, -_c],
                [_b*cosab, _b*np.sqrt(1-cosab**2), 0],
                [_a, 0, _c],
            ])
    elif abs(a - b) < eps:
        if abs(cosac+cosbc) < eps:
            print('Found a B-type base-centred monoclinic cell, special axis: v3')
            parameters['ibrav'] = -13
            _a = a*((1+cosab)*2)**0.5
            _b = a*((1-cosab)*2)**0.5
            _cosac = (a / (1/2*_a)) * cosac
            _c = 2*c
            parameters['a'] = _a
            parameters['b'] = c
            parameters['c'] = _b
            parameters['cosac'] = cosac
            cell_qe = 1/2 * np.array([
                [_a, _b, 0],
                [-_a, _b, 0],
                [_c*_cosac, 0, _c*np.sqrt(1-_cosac**2)],
            ])
        if abs(cosac-cosbc) < eps:
            print('Found a A-type base-centred monoclinic cell, special axis: v3')
            parameters['ibrav'] = 13
            _a = a*((1+cosab)*2)**0.5
            _b = a*((1-cosab)*2)**0.5
            _c = 2*c
            _cosac = (a / (1/2*_a)) * cosac
            parameters['a'] = _a
            parameters['b'] = _c
            parameters['c'] = _b
            parameters['cosab'] = cosac
            cell_qe = 1/2 * np.array([
                [_a, 0, -_b],
                [_c*_cosac, _c*np.sqrt(1-_cosac**2), 0],
                [_a, 0, _b],
            ])
    elif abs(b - c) < eps:
        if abs(cosab+cosac) < eps:
            parameters['ibrav'] = 13
            _a = a*2
            _b = b*((1+cosbc)*2)**0.5
            _c = b*((1-cosbc)*2)**0.5
            parameters['a'] = _b
            parameters['b'] = _a
            parameters['c'] = _c
            parameters['cosab'] = cosac
            cell_qe = 1/2 * np.array([
                [_a, 0, -_c],
                [_b*cosab, _b*np.sqrt(1-cosab**2), 0],
                [_a, 0, _c],
            ])
        if abs(cosac-cosbc) < eps:
            parameters['ibrav'] = 13
            _a = a*2
            _c = b*((1+cosbc)*2)**0.5
            _b = b*((1-cosbc)*2)**0.5
            parameters['a'] = _b
            parameters['b'] = _a
            parameters['c'] = _c
            cell_qe = 1/2 * np.array([
                [_a, 0, -_c],
                [_b*cosab, _b*np.sqrt(1-cosab**2), 0],
                [_a, 0, _c],
            ])
    # --- TRICLINIC CELL --- #
    # triclinic cell if none of the above cases are satisfied.

    else:
        parameters['ibrav'] = 14
        parameters['a'] = a
        parameters['b'] = b
        parameters['c'] = c
        parameters['cosab'] = cosab
        parameters['cosac'] = cosac
        parameters['cosbc'] = cosbc
        sinab = np.sqrt(1-cosab**2)
        cell_qe = np.array([
            [a, 0, 0],
            [b*cosab, b*sinab, 0],
            [c*cosac, c*cosbc*cosac*cosab/sinab, c*np.sqrt(1+2*cosbc*cosac*cosab-cosbc**2-cosac**2-cosab**2)/sinab],
        ])

    return cell_qe, parameters

def get_cellpar(cell):
    """
    Get the cell parameters from the cell according to the following order:
    a, b, c, cosab, cosac, cosbc.
    """

    v1, v2, v3 = [np.array(v) for v in cell]
    a, b, c = [la.norm(v) for v in [v1, v2, v3]]
    cosab, cosac, cosbc = [np.dot(v1, v2) / (a * b), np.dot(v1, v3) / (a * c), np.dot(v2, v3) / (b * c)]

    return a, b, c, cosab, cosac, cosbc

def get_parameters_from_structure(structure, eps=1e-6):
    """
    Get the cell parameters and the cartesian coordinates of the atoms in the Quantum ESPRESSO convention.
    This method will check all the possible cases where the given cell is equivalent to cell in Quantum ESPRESSO convention modulo a rotation or linear transformation.

    Args:
        structure (StructureData): the structure data.
        eps (float): the tolerance for the cell parameters.

    Returns:
        a dictionary containing the cell parameters and the cartesian coordinates of the atoms in the Quantum ESPRESSO convention.
    """

    cell = np.array(structure.cell)
    ase_atoms = structure.get_ase()
    positions = ase_atoms.get_positions()

    bravais_lattice_name = ase_atoms.cell.get_bravais_lattice(eps=eps).name


    v1, v2, v3 = [np.array(v) for v in cell]
    a, b, c = [la.norm(v) for v in [v1, v2, v3]]
    cosab, cosac, cosbc = [np.dot(v1, v2) / (a * b), np.dot(v1, v3) / (a * c), np.dot(v2, v3) / (b * c)]


    if max(a, b, c) < eps or abs(np.dot(np.cross(v1, v2), v3)) < eps:
        raise ValueError("Invalid cell")

    cell_qe, parameters = get_cell_qe_convention(cell, eps=eps)


    # decompose the transformation from cell to cell_qe into a physical rotation (R) and a basis change (T)
    # and apply the rotation to the cartesian coordinates of the atoms
    # The basis change will not affect the cartesian coordinates of the atoms since it just make the atoms
    # outside the new unit cell.

    T = base_transformation(cell, cell_qe)

    ibrav_ase = ibrav_bravais_lattice_map_ase[bravais_lattice_name]
    if parameters['ibrav'] != ibrav_ase:
        raise Warning(f"Found bravais lattice {parameters['ibrav']} is different from ase: {bravais_lattice_name}({ibrav_ase})")

    return parameters, positions

def get_standardized_structure_pymatgen(structure, eps=1e-6):
    """
    Get the standardized structure in the Quantum ESPRESSO convention.
    """

    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    spg_analyzer = SpacegroupAnalyzer(structure.get_pymatgen(), symprec=eps, angle_tolerance=eps)

    primitive_standard_structure = spg_analyzer.get_primitive_standard_structure()

    return primitive_standard_structure

def convert_standardized_structure_pymatgen_to_qe(pym_structure, eps=1e-6):
    """
    Convert the standardized structure in pymatgen to the Quantum ESPRESSO convention.
    """


    cell = pym_structure.lattice.matrix
    a, b, c, cosab, cosac, cosbc = get_cellpar(cell)


    parameters = {
        'ibrav': 0
        }
    cell_qe = None

    transformation_matrix = [
        [1,  0,  0],
        [0,  1,  0],
        [0,  0,  1],
    ]
    # --- CUBIC CELL --- #
    # cubic cell if three vectors have the same length and are mutually orthogonal
    # Note: the cell is allowed to have arbitrary rotation.
    if (
        (max(a, b, c) - min(a, b, c)) <= eps
        and
        max(abs(cosab), abs(cosac), abs(cosbc)) <= eps
        ):
        if not np.allclose(cell/a, np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
            ]), atol=eps):
            raise ValueError("Invalid cell for cubic")

        transformation_matrix = [
            [1,  0,  0],
            [0,  1,  0],
            [0,  0,  1],
        ]
        parameters['ibrav'] = 1
        parameters['a'] = a
        cell_qe = a * np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

    # --- FCC CELL --- #
    # FCC cell if three vectors have the same length and there angles are either 60° or 120°
    # Note: there are 12 nearest neighbours to the origin. All of them are equivalent to the origin.
    # Note: the cell is allowed to have arbitrary rotation.
    elif (
        (max(a, b, c) - min(a, b, c)) <= eps
        and
        max(abs(abs(cosab) - 1/2), abs(abs(cosac) - 1/2), abs(abs(cosbc) - 1/2)) <= eps
    ):
        cell_pattern = cell / (a/2**0.5)

        if not np.allclose(cell_pattern, np.array(
            [
                [ 0, 1, 1],
                [ 1, 0, 1],
                [ 1, 1, 0],
            ]), atol=eps):
            raise ValueError("Invalid cell for FCC")

        transformation_matrix = [
            [1,  0, -1],
            [1,  0,  0],
            [1, -1,  0],
        ]

        _a = a*2**0.5
        parameters['ibrav'] = 2
        parameters['a'] = a

        # pymatgen convention is linked to the QE convention by a following transformation:
        # [1, 0, -1]
        # [1, 0, 0]
        # [1, -1, 0]
        # This won't affect the cartesian coordinates of the atoms.

        cell_qe = _a / 2 * np.array([
            [-1, 0, 1],
            [ 0, 1, 1],
            [-1, 1, 0],
        ])

    # --- BCC CELL --- #
    # BCC cell if three vectors have the same length and there angles are either arccos(1/9) or arccos(-1/9)
    # Note: there are 8 nearest neighbours to the origin. All of them are equivalent to the origin.
    # Note: the cell is allowed to have arbitrary rotation.
    elif (
        (max(a, b, c) - min(a, b, c)) <= eps
        and
        max(abs(abs(cosab) - 1/3), abs(abs(cosac) - 1/3), abs(abs(cosbc) - 1/3)) <= eps
    ):
        cell_pattern = cell / (a/3**0.5)

        if not np.allclose(cell_pattern, np.array(
            [
                [-1,  1,  1],
                [ 1, -1,  1],
                [ 1,  1, -1],
            ]), atol=eps):
            raise ValueError("Invalid cell for BCC")

        transformation_matrix = [
            [1,  1,  1],
            [1,  0,  0],
            [0,  0,  -1],
        ]
        # pymatgen convention is the same as QE convention for ibrav = -3

        _a = a*2/3**0.5
        parameters['ibrav'] = 3
        parameters['a'] = _a
        cell_qe_A = _a / 2* np.array([
            [ -1, 1,  1],
            [ 1, -1,  1],
            [ 1,  1, -1],
        ])
        cell_qe_B = _a / 2* np.array([
            [  1,  1,  1],
            [ -1,  1,  1],
            [ -1, -1,  1],
        ])
    # --- HEXAGONAL CELL --- #
    # Hexagonal cell if two of the three vectors have the same length and the angle between them is 120°
    # Note: the cell is allowed to have arbitrary rotation.
    elif (
        abs(a - b) <= eps
        and
        max(abs(abs(cosab) - 1/2), abs(cosac), abs(cosbc)) <= eps
    ):
        cell_pattern = cell / a

        if not np.allclose(cell_pattern, np.array(
            [
                [ 1/2, -np.sqrt(3)/2, 0  ],
                [1/2, np.sqrt(3)/2, 0],
                [ 0, 0, c/a],
            ]), atol=eps):
            raise ValueError("Invalid cell for HEX")
        # pymatgen convention is linked to the QE convention by a following transformation:
        # [0, 1, 0]
        # [1, 0, 0]
        # [0, 0, 1]
        # This won't affect the cartesian coordinates of the atoms.


        rotation_matrix = SymmOp.from_axis_angle_and_translation(
            axis=[0,0,1], angle=60
            )

        pym_structure.apply_operation(rotation_matrix)

        parameters['ibrav'] = 4
        parameters['a'] = a
        parameters['c'] = c
        cell_qe = a * np.array([
            [1, 0, 0],
            [-1/2, np.sqrt(3)/2, 0],
            [0, 0, c/a],
        ])

    # --- Rhombohedral CELL --- #
    # rhombohedral cell if three vectors have the same length and there angles are the same.
    # Note: unlike Quantum ESPRESSO, thermo_pw.x only accept definition of cosab although in rhombohedral cell although cosab == cosac == cosbc.
    # Note: there are 6 nearest neighbours to the origin. All of them are equivalent to the origin.
    # Note: the cell is allowed to have arbitrary rotation.
    elif (
        (max(a, b, c) - min(a, b, c)) <= eps
        and
        max(cosab, cosac, cosbc) - min(cosab, cosac, cosbc) < eps
    ):
        cell_pattern = cell / a

        tx = np.sqrt((1-cosab)/2)
        ty = np.sqrt((1-cosab)/6)
        tz = np.sqrt((1+2*cosab)/3)

        if not np.allclose(cell_pattern, np.array(
            [
                [0, 2*ty, tz],
                [tx, -ty, tz],
                [-tx, -ty, tz],
            ]), atol=eps):
            raise ValueError("Invalid cell for HEX")

        transformation_matrix = [
            [0,  1,  0],
            [1,  0,  0],
            [0,  0,  1],
        ]

        parameters['ibrav'] = 5
        parameters['a'] = a
        parameters['cosab'] = cosab

        cell_qe = a * np.array([
            [tx, -ty, tz],
            [0, 2*ty, tz],
            [-tx, -ty, tz],
        ])
    # --- TETRAGONAL CELL --- #
    # Tetragonal cell if two of the three vectors have the same length and the angle between them is 90°
    # Note: the cell is allowed to have arbitrary rotation.
    elif (
        abs(a - b) <= eps
        and
        max(cosab, cosac, cosbc) - min(cosab, cosac, cosbc) < eps
        ):
        cell_pattern = cell / a

        if not np.allclose(cell_pattern, np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, c/a],
            ]), atol=eps):
            raise ValueError("Invalid cell for TET")

        parameters['ibrav'] = 6
        parameters['a'] = a
        parameters['c'] = c
        cell_qe = a * np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, c/a],
        ])

    # --- BODY-CENTRED TETRAGONAL CELL --- #
    # body-centred tetragonal cell if three vectors have the following pattern:
    # [1, 0, 0]
    # [0, 1, 0]
    # [0, 0, c/a]
    elif (
        max(a, b, c) - min(a, b, c) < eps
        and
        (
            abs(cosab - cosbc) < eps and abs(abs(cosab) + abs(cosbc) - cosac - 1) < eps
            or
            abs(cosab - cosac) < eps and abs(abs(cosab) + abs(cosac) - cosbc - 1) < eps
            or
            abs(cosbc - cosac) < eps and abs(abs(cosbc) + abs(cosac) - cosab - 1) < eps
        )
    ):
        _a = ((a*a - a*b*cosab)/4)**0.5
        _c = ((a*a + a*b*cosab)/2)**0.5

        cell_pattern = cell / _a

        if not np.allclose(cell_pattern, np.array(
            [
                [-1, 1, _c/_a],
                [1, -1, _c/_a],
                [1, 1, -_c/_a],
            ]), atol=eps):
            raise ValueError("Invalid cell for BCT")

        # pymatgen convention is linked to the QE convention by a following transformation:
        # [1, 0, 0]
        # [0, 1, 0]
        # [0, 0, c/a]
        # This won't affect the cartesian coordinates of the atoms.

        transformation_matrix = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, -1],
        ]
        parameters['ibrav'] = 7
        parameters['a'] = _a*2
        parameters['c'] = _c*2
        cell_qe = _a * np.array([
            [1, -1, _c/_a],
            [1, 1, _c/_a],
            [-1, -1, _c/_a],
        ])

    # --- ORTHORHOMBIC CELL --- #
    # orthorhombic cell if three vectors have the same length and are mutually orthogonal
    # Note: the cell is allowed to have arbitrary rotation.
    elif (
        max(abs(cosab), abs(cosac), abs(cosbc)) <= eps
    ):
        if not np.allclose(cell, np.array([
            [a, 0, 0],
            [0, b, 0],
            [0, 0, c],
        ]), atol=eps):
            raise ValueError("Invalid cell for ORC")

        parameters['ibrav'] = 8
        parameters['a'] = a
        parameters['b'] = b
        parameters['c'] = c
        cell_qe = np.array([
            [a, 0, 0],
            [0, b, 0],
            [0, 0, c],
        ])
    elif (
        max(abs(a - b), abs(cosac), abs(cosbc)) < eps
    ):
        _a = ((a**2 + a*b*cosab)/2)**0.5
        _b = ((a**2 - a*b*cosab)/2)**0.5

        if not np.allclose(cell, np.array([
            [_a,-_b, 0],
            [_a, _b, 0],
            [ 0,  0, c],
        ]), atol=eps):
            raise ValueError("Invalid cell for ORH")

        transformation_matrix = [
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1],
        ]
        parameters['ibrav'] = 9
        parameters['a'] = _a*2
        parameters['b'] = _b*2
        parameters['c'] = c
        cell_qe = {
            9: np.array([
            [_a, _b, 0],
            [-_a, _b, 0],
            [ 0,  0, c],
        ]),
            -9: np.array([
            [_a, -_b, 0],
            [_a, _b, 0],
            [ 0,  0, -c],
        ])}
    elif (
        max(a, b, c) - min(a, b, c) < eps
        and
        abs(cosab+cosbc+cosac+1) < eps
    ):
        _a = ((b**2 + b*c*cosbc)/2)**0.5
        _b = ((a**2 + a*c*cosac)/2)**0.5
        _c = ((c**2 + a*b*cosab)/2)**0.5

        if not np.allclose(cell, np.array([
            [-_a, _b, _c],
            [_a, -_b, _c],
            [_a, _b, -_c],
        ]), atol=eps):
            raise ValueError("Invalid cell for ORT")

        transformation_matrix = [
            [1, 1, 1],
            [1, 0, 0],
            [0, 0, -1],
        ]
        parameters['ibrav'] = 11
        parameters['a'] = _a*2
        parameters['b'] = _b*2
        parameters['c'] = _c*2
        cell_qe = np.array([
            [_a, _b, _c],
            [-_a, _b, _c],
            [-_a, -_b, _c],
        ])
    elif (
        max(abs(cosab), abs(cosbc)) < eps
        ):
        _cy = cosac*c
        _cz = -c*np.sqrt(1-cosac**2)

        if not np.allclose(cell, np.array([
            [0, a, 0],
            [b, 0, 0],
            [0, _cy, _cz],
        ]), atol=eps):
            raise ValueError("Invalid cell for MCL")

        rotation_matrix = SymmOp.from_axis_angle_and_translation(
            axis=[0,0,1], angle=-90
            )

        pym_structure.apply_operation(rotation_matrix)

        transformation_matrix = [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]
        parameters['ibrav'] = -12
        parameters['a'] = a
        parameters['b'] = b
        parameters['c'] = c
        parameters['cosac'] = cosac
        cell_qe = np.array([
            [a, 0, 0],
            [0, b, 0],
            [_cy, 0, _cz],
        ])
    elif (
        abs(a - b) < eps
        and
        abs(cosac+cosbc) < eps
    ):
        _a = ((a**2 + a*b*cosab)/2)**0.5
        _b = ((a**2 - a*b*cosab)/2)**0.5
        _cosac = a*cosac/_b
        _cy = c*_cosac
        _cz = -c*np.sqrt(1-_cosac**2)

        if not np.allclose(cell, np.array([
            [_a, _b, 0],
            [_a, -_b, 0],
            [0, _cy, _cz],
        ]), atol=eps):
            raise ValueError("Invalid cell for MCLC")

        rotation_matrix = SymmOp.from_axis_angle_and_translation(
            axis=[0,0,1], angle=90
            )

        pym_structure.apply_operation(rotation_matrix)

        transformation_matrix = [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1],
        ]
        parameters['ibrav'] = -13
        parameters['a'] = _b*2
        parameters['b'] = _a*2
        parameters['c'] = c*2
        parameters['cosac'] = _cosac
        cell_qe = np.array([
            [_a, _b, 0],
            [-_a, _b, 0],
            [_cy, 0, _cz],
        ])
    else:
        sinab = np.sin(np.arccos(cosab))
        if cell[0][0] < 0:
            transformation_matrix = [
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
            pym_structure.make_supercell(transformation_matrix)
            cell = pym_structure.lattice.matrix

        if cell[1][1] < 0:
            transformation_matrix = [
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1],
            ]
            pym_structure.make_supercell(transformation_matrix)
            cell = pym_structure.lattice.matrix

        # if cell[2][0] < 0:
        #     transformation_matrix = [
        #         [1, 0, 0],
        #         [0, 1, 0],
        #         [0, 0, -1],
        #     ]
        #     pym_structure.make_supercell(transformation_matrix)
        #     cell = pym_structure.lattice.matrix

        if np.allclose(cell, np.array([
            [a, 0, 0],
            [b*cosab, b*sinab, 0],
            [c*cosac,  c*(cosbc-cosac*cosab)/sinab, c*np.sqrt( 1 + 2*cosbc*cosac*cosab- cosbc**2-cosac**2-cosab**2 )/sinab],
        ]), atol=eps):

            parameters['ibrav'] = 14
            parameters['a'] = a
            parameters['b'] = b
            parameters['c'] = c
            parameters['cosab'] = cosab
            parameters['cosbc'] = cosbc
            parameters['cosac'] = cosac
            cell_qe = np.array([
                [a, 0, 0],
                [b*cosab, b*sinab, 0],
                [c*cosac,  c*(cosab-cosac*cosab)/sinab, c*np.sqrt( 1 + 2*cosbc*cosac*cosab- cosbc**2-cosac**2-cosab**2 )/sinab],
            ])

        else:
            cell_str = np.array2string(cell, precision=6, separator=',', suppress_small=True)
            pattern_str = np.array2string(np.array([
                [a, 0, 0],
                [b*cosab, b*sinab, 0],
                [c*cosac,  c*(cosbc-cosac*cosab)/sinab, c*np.sqrt( 1 + 2*cosbc*cosac*cosab- cosbc**2-cosac**2-cosab**2 )/sinab],
            ]), precision=6, separator=',', suppress_small=True)
            raise ValueError(f"Cell \n{cell_str} does not match the expected pattern \n{pattern_str}. Please check the cell parameters or use ibrav = 0.")

    new_pym_structure = pym_structure.make_supercell(transformation_matrix)

    return new_pym_structure, parameters

def check_conversion(structure, log=print):
    cell = np.array(structure.cell)
    cellpar = get_cellpar(structure.cell)
    cell_qe, parameters = get_cell_qe_convention(structure.cell, eps=1e-6)
    cellpar_qe = get_cellpar(cell_qe)
    ibrav_ase = get_ibrav_ase(structure)
    log(f'Example: {ibrav_bravais_lattice_map_qe[ibrav_ase]} [{parameters["ibrav"]}]')

    log('Cell from StructureData',)
    log(np.array2string(
        cell, precision=6, separator=',',
        suppress_small=True))

    log('Cell converted to Quantum ESPRESSO convention',)
    log(np.array2string(
        cell_qe, precision=6, separator=',',
        suppress_small=True))
    log(f'Parameters: {parameters}')
    log(f'ibrav_ase: {ibrav_ase}')
    log(f'Cell parameters: a, b, c, cosab, cosac, cosbc')
    log(f'                 {cellpar}')
    log(f'Cell parameters (QE): a, b, c, cosab, cosac, cosbc')
    log(f'                     {cellpar_qe}')
    log('-'*100)

    T = base_transformation(structure.cell, cell_qe)

    log('It requires a basis change to convert the cell to Quantum ESPRESSO convention',)
    log(np.array2string(
        T, precision=6, separator=',',
        suppress_small=True)
        )
    log(f'Determinant of T: {np.linalg.det(T)}')

    reconstructed_cell_qe = np.dot(T, cell.T).T
    restored_cell = np.dot(np.linalg.inv(T), cell_qe).T

    log('reconstructed cell',)
    log(np.array2string(
        reconstructed_cell_qe, precision=6, separator=',',
        suppress_small=True))
    log('restored cell',)
    log(np.array2string(
        restored_cell, precision=6, separator=',',
        suppress_small=True))
    log(f'Difference between reconstructed cell: {np.linalg.norm(reconstructed_cell_qe - cell_qe)}')
    log(f'Difference between restored cell: {np.linalg.norm(restored_cell - cell)}')

