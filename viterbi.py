import numpy as np
import math

def viterbi_normal(A, B, pi, omega):
    """
    Implementation of the Viterbi algorithm for finding the most likely state sequence in an HMM.
    This is the standard version without logarithms.
    
    Parameters:
    A -- n x n transition matrix where A[i,j] is the probability of transitioning from state i to state j
    B -- n x m emission matrix where B[i,j] is the probability of emitting symbol j from state i
    pi -- Initial state probabilities (vector of length n)
    omega -- Observation sequence (list of indices corresponding to observed symbols)
    
    Returns:
    tuple -- (state_sequence, max_probability)
        state_sequence: The most likely sequence of states that produced the observation sequence
        max_probability: The probability of the most likely path
    """
    n = A.shape[0]  # Number of states
    T = len(omega)  # Length of observation sequence
    
    # Initialize
    score = np.zeros((n, T))
    backpointer = np.zeros((n, T), dtype=int)
    
    # Initialization (t=0)
    for j in range(n):
        score[j, 0] = pi[j] * B[j, omega[0]]
    
    # Forward pass
    for t in range(1, T):
        for j in range(n):
            # Calculate all possible transitions to state j
            temp_scores = [score[k, t-1] * A[k, j] * B[j, omega[t]] for k in range(n)]
            
            # Find the maximum score and its corresponding state
            score[j, t] = max(temp_scores)
            backpointer[j, t] = np.argmax(temp_scores)
    
    # Termination: find the state with highest probability at the final time step
    max_score = max(score[:, T-1])
    last_state = np.argmax(score[:, T-1])
    
    # Path retrieval (backtracking)
    state_indices = [0] * T
    state_indices[T-1] = last_state
    
    for t in range(T-1, 0, -1):
        state_indices[t-1] = backpointer[state_indices[t], t]
    
    # Convert to 1-based indices for output
    state_indices = [i+1 for i in state_indices]
    
    return state_indices, max_score

def viterbi_log(A, B, pi, omega):
    """
    Implementation of the Viterbi algorithm for finding the most likely state sequence in an HMM.
    This version uses logarithms to avoid underflow for long sequences.
    
    Parameters:
    A -- n x n transition matrix where A[i,j] is the probability of transitioning from state i to state j
    B -- n x m emission matrix where B[i,j] is the probability of emitting symbol j from state i
    pi -- Initial state probabilities (vector of length n)
    omega -- Observation sequence (list of indices corresponding to observed symbols)
    
    Returns:
    tuple -- (state_sequence, max_log_probability)
        state_sequence: The most likely sequence of states that produced the observation sequence
        max_log_probability: The log probability of the most likely path
    """
    n = A.shape[0]  # Number of states
    T = len(omega)  # Length of observation sequence
    
    # Initialize with logarithms to avoid underflow
    score = np.zeros((n, T))
    backpointer = np.zeros((n, T), dtype=int)
    
    # Initialization (t=0)
    for j in range(n):
        score[j, 0] = math.log(pi[j]) + math.log(B[j, omega[0]])
    
    # Forward pass
    for t in range(1, T):
        for j in range(n):
            # Calculate all possible transitions to state j
            temp_scores = [score[k, t-1] + math.log(A[k, j]) + math.log(B[j, omega[t]]) for k in range(n)]
            
            # Find the maximum score and its corresponding state
            score[j, t] = max(temp_scores)
            backpointer[j, t] = np.argmax(temp_scores)
    
    # Termination: find the state with highest probability at the final time step
    max_score = max(score[:, T-1])
    last_state = np.argmax(score[:, T-1])
    
    # Path retrieval (backtracking)
    state_indices = [0] * T
    state_indices[T-1] = last_state
    
    for t in range(T-1, 0, -1):
        state_indices[t-1] = backpointer[state_indices[t], t]
    
    # Convert to 1-based indices for output
    state_indices = [i+1 for i in state_indices]
    
    return state_indices, max_score

def run_example(A, B, pi, observation_sequence, output_states_map=None, use_logarithm=False, print_observation=True):
    """
    Run the Viterbi algorithm on an example and print the results.
    
    Parameters:
    A -- Transition matrix
    B -- Emission matrix
    pi -- Initial state probabilities
    observation_sequence -- Sequence of observation symbols
    output_states_map -- Optional mapping to convert state indices to state names
    use_logarithm -- If True, use the logarithmic version of the Viterbi algorithm
    """
    # Convert observation sequence to 0-based indices for processing
    omega = [i-1 for i in observation_sequence]
    
    if use_logarithm:
        state_indices, max_score = viterbi_log(A, B, pi, omega)
        print(f"Using logarithmic Viterbi algorithm")
    else:
        state_indices, max_score = viterbi_normal(A, B, pi, omega)
        print(f"Using standard Viterbi algorithm")
    
    if print_observation:
        print(f"Observation sequence: {observation_sequence}")
        print(f"Most likely state sequence (indices): {state_indices}")
    
    if output_states_map:
        state_sequence = [output_states_map[i] for i in state_indices]
        if print_observation:
            print(f"Most likely state sequence (names): {state_sequence}")
    
    if use_logarithm:
        print(f"Maximum log probability: {max_score}")
        print(f"Maximum probability: {math.exp(max_score)}")
    else:
        print(f"Maximum probability: {max_score}")
    
    print("---")
    
    return state_indices, max_score