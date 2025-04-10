import numpy as np
from viterbi import viterbi_normal, viterbi_log, run_example

def example_4_1(use_logarithm=False):
    """
    Run the Viterbi algorithm on Example 4.1 from the notes.
    States: {Cold, Hot} with σ(Cold)=1, σ(Hot)=2
    Outputs: {N, D} with ω(N)=1, ω(D)=2
    """
    # Define the HMM parameters for Example 4.1
    A = np.array([
        [0.7, 0.3],  # Transition from Cold to Cold, Cold to Hot
        [0.25, 0.75]   # Transition from Hot to Cold, Hot to Hot
    ])
    
    B = np.array([
        [0.8, 0.2],  # Emission from Cold to N, Cold to D
        [0.3, 0.7]   # Emission from Hot to N, Hot to D
    ])
    
    pi = np.array([0.45, 0.55])  # Initial probability of Cold, Hot
    
    # Define state and observation mappings
    states_map = {1: "Cold", 2: "Hot"}
    obs_map = {1: "N", 2: "D"}
    
    # Test cases from the homework
    print("========== Example 4.1 Tests ==========")
    
    # Test case 1: NNND
    # From the notes, we expect state sequence (Cold, Cold, Cold, Hot) = (1, 1, 1, 2)
    observation = [1, 1, 1, 2]  # NNND
    print("Test case 1: NNND")
    run_example(A, B, pi, observation, states_map, use_logarithm=use_logarithm)
    
    # Test case 2: NNNDN
    observation = [1, 1, 1, 2, 1]  # NNNDN
    print("Test case 2: NNNDN")
    run_example(A, B, pi, observation, states_map, use_logarithm=use_logarithm)
    
    # Test case 3: NNDNN
    observation = [1, 1, 2, 1, 1]  # NNDNN
    print("Test case 3: NNDNN")
    run_example(A, B, pi, observation, states_map, use_logarithm=use_logarithm)
    
    # Test case 4: NNNDNDDN
    observation = [1, 1, 1, 2, 1, 2, 2, 1]  # NNNDNDDN
    print("Test case 4: NNNDNDDN")
    run_example(A, B, pi, observation, states_map, use_logarithm=use_logarithm)
    
    # Test case 5: Long sequence 1
    print("Test Case 5: 300N 300D 300N 300D")
    # Create a sequence of length 1200 using the pattern N...ND...DN...ND...D (300 each)
    observation = []
    observation.extend([1] * 300)  # 300 N
    observation.extend([2] * 300)  # 300 D
    observation.extend([1] * 300)  # 300 N
    observation.extend([2] * 300)  # 300 D
    state_indices, max_score = run_example(A, B, pi, observation, states_map, use_logarithm=use_logarithm, print_observation=False)
    print(len(state_indices), "states") 
    print(f"States q1-q5: {[states_map[idx] for idx in state_indices[:5]]}")
    print(f"States q300-q304: {[states_map[idx] for idx in state_indices[299:304]]}")
    print(f"States q600-q604: {[states_map[idx] for idx in state_indices[599:604]]}")
    print(f"States q900-q904: {[states_map[idx] for idx in state_indices[899:904]]}")
    print(f"States q1196-q1200: {[states_map[idx] for idx in state_indices[1195:1200]]}")
    print("----------------------------------------------")

    # Test case 6: Long sequence 2
    print("Test Case 6: 500N 500D 500N 500D")
    # Create a sequence of length 2000 using the pattern N...ND...DN...ND...D (500 each)
    observation = []
    observation.extend([1] * 500)  # 500 N
    observation.extend([2] * 500)  # 500 D
    observation.extend([1] * 500)  # 500 N
    observation.extend([2] * 500)  # 500 D
    state_indices, max_score = run_example(A, B, pi, observation, states_map, use_logarithm=use_logarithm, print_observation=False) 
    print(f"States q1-q5: {[states_map[idx] for idx in state_indices[:5]]}")
    print(f"States q500-q504: {[states_map[idx] for idx in state_indices[499:504]]}")
    print(f"States q1000-q1004: {[states_map[idx] for idx in state_indices[999:1004]]}")
    print(f"States q1500-q1504: {[states_map[idx] for idx in state_indices[1499:1504]]}")
    print(f"States q1996-q2000: {[states_map[idx] for idx in state_indices[1995:2000]]}")
    print("----------------------------------------------")

    # Test case 7: Long sequence 2
    print("Test Case 7: 500N 500D 500N 500D NNND")
    # Create a sequence of length 2004 using the pattern N...ND...DN...ND...D NNND
    observation = []
    observation.extend([1] * 500)  # 500 N
    observation.extend([2] * 500)  # 500 D
    observation.extend([1] * 500)  # 500 N
    observation.extend([2] * 500)  # 500 D
    observation.extend([1, 1, 1, 2]) # NNND
    state_indices, max_score = run_example(A, B, pi, observation, states_map, use_logarithm=use_logarithm, print_observation=False) 
    print(f"States q1-q5: {[states_map[idx] for idx in state_indices[:5]]}")
    print(f"States q500-q504: {[states_map[idx] for idx in state_indices[499:504]]}")
    print(f"States q1000-q1004: {[states_map[idx] for idx in state_indices[999:1004]]}")
    print(f"States q1500-q1504: {[states_map[idx] for idx in state_indices[1499:1504]]}")
    print(f"States q1999-q2004: {[states_map[idx] for idx in state_indices[1998:2004]]}")
    print("----------------------------------------------")

def dna_example(use_logarithm=False):
    """
    Run the Viterbi algorithm on the DNA example from the homework.
    States: {H, L} with σ(H)=1, σ(L)=2
    Outputs: {A, C, G, T} with ω(A)=1, ω(C)=2, ω(G)=3, ω(T)=4
    """
    # Define the HMM parameters for the DNA example
    A = np.array([
        [0.5, 0.5],  # Transition from H to H, H to L
        [0.4, 0.6]   # Transition from L to H, L to L
    ])
    
    B = np.array([
        [0.2, 0.3, 0.3, 0.2],  # Emission from H to A, C, G, T
        [0.3, 0.2, 0.2, 0.3]   # Emission from L to A, C, G, T
    ])
    
    pi = np.array([0.5, 0.5])  # Initial probability of H, L
    
    # Define state and observation mappings
    states_map = {1: "H", 2: "L"}
    obs_map = {1: "A", 2: "C", 3: "G", 4: "T"}
    
    print("========== DNA Example Tests ==========")
    
    # Convert DNA sequence to observation indices
    def dna_to_indices(dna_sequence):
        mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
        return [mapping[nucleotide] for nucleotide in dna_sequence]
    
    # Test case: GGCACTGAA
    dna_sequence = "GGCACTGAA"
    observation = dna_to_indices(dna_sequence)
    print(f"DNA sequence: {dna_sequence}")
    run_example(A, B, pi, observation, states_map, use_logarithm=use_logarithm)
    
    # Test case: GAGATATACATAGAATTACG
    dna_sequence = "GAGATATACATAGAATTACG"
    observation = dna_to_indices(dna_sequence)
    print(f"DNA sequence: {dna_sequence}")
    run_example(A, B, pi, observation, states_map, use_logarithm=use_logarithm)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Viterbi algorithm tests')
    parser.add_argument('--test', type=str, default='all', 
                        choices=['all', 'example4.1'],
                        help='Specify which test to run')
    parser.add_argument('--algorithm', type=str, default='normal', 
                        choices=['normal', 'log'],
                        help='Specify which algorithm version to use (normal or logarithmic)')
    
    args = parser.parse_args()
    use_logarithm = (args.algorithm == 'log')
    
    if args.test == 'all':
        example_4_1(use_logarithm)
        print("\n")
        dna_example(use_logarithm)
        print("\n")
    elif args.test == 'example4.1':
        example_4_1(use_logarithm)
    elif args.test == 'dna':
        dna_example(use_logarithm)
    