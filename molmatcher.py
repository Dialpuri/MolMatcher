"""
MolMatcher - Jordan Dialpuri 2024 ©
"""

from typing import Tuple, List
import gemmi
import numpy as np
import networkx as nx

def create_single_residue(structure: gemmi.Structure) -> gemmi.Structure:
    """
    Creates a single residue structure where all atoms are in one residue.
    :param structure: Structure to modify
    :return: Single residue structure
    """
    os = gemmi.Structure()
    om = gemmi.Model(structure[0].name)
    oc = gemmi.Chain("A")
    or_ = gemmi.Residue()
    or_.seqid = gemmi.SeqId("1")
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    or_.add_atom(atom)
    oc.add_residue(or_)
    om.add_chain(oc)
    os.add_model(om)
    return os


def move_to_centroid(structure: gemmi.Structure) -> gemmi.Structure:
    """
    Move the structure to the center of the cell.
    :param structure: Structure to move
    :return: Modified structure
    """
    centroid = np.mean([a.pos.tolist() for a in structure[0][0][0]], axis=0)
    center = np.array([50,50,50])
    delta = center-centroid
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.pos += gemmi.Position(*delta)
    return structure


def get_adjacency_matrix(structure) -> Tuple[np.ndarray, List[str]]:
    """
    Converts the structure into an adjacency matrix.
    :param structure: Single residue structure
    :return: Tuple of adjacency matrix and labels
    """
    ns = gemmi.NeighborSearch(structure[0], structure.cell, 2).populate()
    data = set()
    for chain in structure[0]:
        for residue_index, residue in enumerate(chain):
            for atom_index, atom in enumerate(residue):
                nearby_atoms = ns.find_atoms(atom.pos)

                for near_atom in nearby_atoms:
                    dist = (atom.pos - near_atom.pos).length()
                    if dist < 0.5 or dist > 2.0:
                        continue

                    linear_key = (atom_index, near_atom.atom_idx)
                    reverse_key = (linear_key[1], linear_key[0])
                    if reverse_key not in data:
                        data.add(linear_key)

    Xs, Ys = list(zip(*data))
    max_length = max(Xs + Ys)+1

    # Create adj mat
    adjacency_matrix = np.zeros((max_length, max_length))

    for linear_key in data:
        adjacency_matrix[linear_key[0], linear_key[1]] = 1

    keys = [structure[0][0][0][i].element.name for i in range(0, max_length)]

    return adjacency_matrix, keys

def label(labelled_structure: gemmi.Structure, unlabelled_structure: gemmi.Structure) -> gemmi.Structure:
    """
    Labels the unlabelled structure according to the given labelled structure.
    :param labelled_structure: Structure with atom names
    :param unlabelled_structure: Structure without atom names
    :return: Structure with atom names
    """
    cell = gemmi.UnitCell(100,100,100,90,90,90) # If any residue is > 100A in length (that's crazy) then bump this

    labelled_structure = create_single_residue(labelled_structure)
    labelled_structure.cell = cell
    unlabelled_structure.cell = cell

    labelled_structure = move_to_centroid(labelled_structure)
    unlabelled_structure = move_to_centroid(unlabelled_structure)

    l, kl = get_adjacency_matrix(labelled_structure)
    s, ks = get_adjacency_matrix(unlabelled_structure)

    # Create Networks
    G1 = nx.from_numpy_array(l)
    G2 = nx.from_numpy_array(s)

    # Assign label to nodes where labels are the element
    for i, node in enumerate(G1.nodes):
        G1.nodes[node]['label'] = kl[i]

    for i, node in enumerate(G2.nodes):
        G2.nodes[node]['label'] = ks[i]

    # Fun graph matching code (don't mess)
    def check_label_match(mapping):
        for n2, n1 in mapping.items():
            if G2.nodes[n2]['label'] != G1.nodes[n1]['label']:
                return False
        return True

    GM = nx.isomorphism.GraphMatcher(G2, G1)
    if not GM.is_isomorphic():
        raise RuntimeError("GraphMatcher failed: no match found")

    all_mappings = list(GM.isomorphisms_iter())
    if not all_mappings:
        raise RuntimeError("No isomorphisms found")

    label_matching_mappings = [mapping for mapping in all_mappings if check_label_match(mapping)]
    if not label_matching_mappings:
        raise RuntimeError("No label matching mappings found")

    node_mapping = label_matching_mappings[0]
    for atom_index, atom in enumerate(unlabelled_structure[0][0][0]):
        atom.name = labelled_structure[0][0][0][node_mapping[atom_index]].name

    return unlabelled_structure

def main():
    reference_structure = gemmi.read_structure("labelled.pdb")
    unlabelled_structure = gemmi.read_structure("stripped.pdb")

    labelled_structure = label(reference_structure, unlabelled_structure)
    labelled_structure.write_pdb("relabelled.pdb")


if __name__ == '__main__':
    main()