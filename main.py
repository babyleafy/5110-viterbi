import numpy as np
import math
import argparse
from viterbi import viterbi_normal, viterbi_log, run_example

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the Viterbi algorithm for an HMM.')
    
    parser.add_argument('--test', type=str, default='all', 
                      choices=['all', 'example4.1', 'dna'],
                      help='Specify which test to run')
    
    parser.add_argument('--algorithm', type=str, default='normal', 
                        choices=['normal', 'log'],
                        help='Specify which algorithm version to use (normal or logarithmic)')
    
    parser.add_argument('--transition_matrix', '-a', type=str,
                      help='Path to the transition matrix A file (n x n)')
    
    parser.add_argument('--emission_matrix', '-b', type=str,
                      help='Path to the emission matrix B file (n x m)')
    
    parser.add_argument('--initial_probabilities', '-p', type=str,
                      help='Path to the initial probabilities π file')
    
    parser.add_argument('--observations', '-o', type=str,
                      help='Path to the observation sequence file')
    
    parser.add_argument('--debug', '-d', action='store_true',
                      help='Enable debug mode')
    
    return parser.parse_args()

def load_matrix(filepath):
    """Load a matrix from a file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        matrix = []
        for line in lines:
            row = [float(x) for x in line.strip().split()]
            matrix.append(row)
        return np.array(matrix)

def load_vector(filepath):
    """Load a vector from a file."""
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        vector = [float(x) for x in line.split()]
        return np.array(vector)

def load_observations(filepath):
    """Load an observation sequence from a file."""
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        # Observations are 1-indexed in the problem, but 0-indexed in our algorithm
        # We'll convert them back to 1-indexed when outputting
        observations = [int(x) for x in line.split()]
        return observations

def create_observation_sequence(pattern, length):
    """
    Create a sequence of specified length using the given pattern.
    
    Parameters:
    pattern -- Dictionary describing the pattern {'symbols': list, 'repeat': int}
    length -- Total length of the sequence
    
    Returns:
    list -- The generated observation sequence
    """
    result = []
    symbols = pattern['symbols']
    repeat = pattern['repeat']
    
    for _ in range(length // len(symbols)):
        result.extend(symbols)
    
    return result

def get_weather_example_parameters():
    """Get parameters for the weather (Cold/Hot) example from Example 4.1"""
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
    
    return A, B, pi, states_map, obs_map

def get_dna_example_parameters():
    """Get parameters for the DNA (H/L) example"""
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
    
    return A, B, pi, states_map, obs_map

def run_tests(args):
    """Run the specified tests based on command line arguments."""
    use_logarithm = (args.algorithm == 'log')
    
    if args.transition_matrix and args.emission_matrix and args.initial_probabilities and args.observations:
        # Load the matrices and vectors from files
        A = load_matrix(args.transition_matrix)
        B = load_matrix(args.emission_matrix)
        pi = load_vector(args.initial_probabilities)
        omega = load_observations(args.observations)
        
        # Run the Viterbi algorithm
        state_indices, max_score = run_example(A, B, pi, omega, use_logarithm=use_logarithm, debug=args.debug)
        
        return state_indices, max_score
    
    # For specific test cases
    if args.test in ['all', 'example4.1']:
        # Define the HMM parameters for Example 4.1
        A, B, pi, states_map, obs_map = get_weather_example_parameters()
        
        if args.test in ['all', 'example4.1']:
            print("\n========== Example 4.1 Tests ==========")
            
            # Test case 1: NNND
            observation = [1, 1, 1, 2]  # NNND
            print("\nTest case 1: NNND")
            run_example(A, B, pi, observation, states_map, use_logarithm)
            
            # Test case 2: NNNDN
            observation = [1, 1, 1, 2, 1]  # NNNDN
            print("\nTest case 2: NNNDN")
            run_example(A, B, pi, observation, states_map, use_logarithm)
            
            # Test case 3: NNDNN
            observation = [1, 1, 2, 1, 1]  # NNDNN
            print("\nTest case 3: NNDNN")
            run_example(A, B, pi, observation, states_map, use_logarithm)
            
            # Test case 4: NNNDNDDN
            observation = [1, 1, 1, 2, 1, 2, 2, 1]  # NNNDNDDN
            print("\nTest case 4: NNNDNDDN")
            run_example(A, B, pi, observation, states_map, use_logarithm)
    
    # Run the DNA tests
    if args.test in ['all', 'dna']:
        A, B, pi, states_map, obs_map = get_dna_example_parameters()
        
        def dna_to_indices(dna_sequence):
            """Convert DNA sequence to observation indices."""
            mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
            return [mapping[nucleotide] for nucleotide in dna_sequence]
        
        if args.test in ['all', 'dna']:
            print("\n========== DNA Example Tests ==========")
            
            # Test case: GGCACTGAA
            dna_sequence = "GGCACTGAA"
            observation = dna_to_indices(dna_sequence)
            print(f"\nDNA sequence: {dna_sequence}")
            run_example(A, B, pi, observation, states_map, use_logarithm)
            
            # Test case: GAGATATACATAGAATTACG
            dna_sequence = "GAGATATACATAGAATTACG"
            observation = dna_to_indices(dna_sequence)
            print(f"\nDNA sequence: {dna_sequence}")
            run_example(A, B, pi, observation, states_map, use_logarithm)

def main():
    """Main function."""
    args = parse_arguments()
    run_tests(args)

if __name__ == "__main__":
    main()