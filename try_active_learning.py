import data_generation as dg
import pc_algorithm as pc_a
from active_learning import Experiment
import numpy as np
import pandas as pd

# remember to change committee size
def get_metrics(num_nodes: int, committee_size: int, epochs: int) -> list[list]:
    G = dg.create_dag(n=num_nodes, expected_degree=3)
    pcdag = pc_a.pc(G)
    experiment = Experiment(5, 5)

    true_DAG, DAG = experiment.random_dag_from_pcdag(pcdag)

    qbc_hamming, qbc_num_interv = experiment.qbc(
        epochs=epochs, committee_size=committee_size, pcdag=pcdag,
        true_causal_dag=true_DAG, true_causal_graph=DAG,
        data=dg.generate_data(DAG), k=num_nodes, _lambda=0.5
    )

    rand_hamming, rand_num_interv = experiment.random_design(
        pcdag=pcdag, true_causal_graph=DAG,
        true_causal_dag=true_DAG, data=dg.generate_data(DAG), k=num_nodes
    )

    rand_adv_hamming, rand_adv_num_interv = experiment.random_adv_design(
        pcdag=pcdag, true_causal_graph=DAG,
        true_causal_dag=true_DAG, data=dg.generate_data(DAG), k=num_nodes
    )

    return [qbc_hamming, rand_hamming, rand_adv_hamming]

# over some amount of tries, gets average hamming distance, number of interventions for a given graph size (# vertices)
def get_average_metrics(num_nodes: int, num_iterations: int, committee_size: int, epochs: int) -> list[int]:
    df = pd.DataFrame([get_metrics(num_nodes, committee_size, epochs) for _ in range(num_iterations)])
    return df.mean(axis=0)

# Pad the lists to ensure they all have the same length
def pad_list(lst, length):
    return lst + [0.0] * (length - len(lst))

def get_average_hamming_distances(num_nodes: int, num_iterations: int, committee_size: int, epochs: int):
    qbc_hamming_results = []
    rand_hamming_results = []
    rand_adv_hamming_results = []

    max_length = 0  # Track the maximum length of the hamming distance lists

    # Collect results and determine the max length
    for _ in range(num_iterations):
        qbc_hamming, rand_hamming, rand_adv_hamming = get_metrics(num_nodes, committee_size, epochs)
        qbc_hamming_results.append(qbc_hamming)
        rand_hamming_results.append(rand_hamming)
        rand_adv_hamming_results.append(rand_adv_hamming)

        # Update the max_length for padding later
        #max_length = max(max_length, len(qbc_hamming), len(rand_hamming), len(rand_adv_hamming))



    qbc_pad = max(len(ls) for ls in qbc_hamming_results)
    rand_pad = max(len(ls) for ls in rand_hamming_results)
    rand_adv_pad = max(len(ls) for ls in rand_adv_hamming_results)

    # Pad each list to the maximum length
    qbc_hamming_results = [pad_list(lst, qbc_pad) for lst in qbc_hamming_results]
    rand_hamming_results = [pad_list(lst, rand_pad) for lst in rand_hamming_results]
    rand_adv_hamming_results = [pad_list(lst, rand_adv_pad) for lst in rand_adv_hamming_results]

    # Convert to numpy arrays for easy computation
    qbc_hamming_results = np.array(qbc_hamming_results)
    rand_hamming_results = np.array(rand_hamming_results)
    rand_adv_hamming_results = np.array(rand_adv_hamming_results)

    # Calculate the mean for each index across all iterations (ignoring NaN values)
    qbc_hamming_mean = np.nanmean(qbc_hamming_results, axis=0)
    rand_hamming_mean = np.nanmean(rand_hamming_results, axis=0)
    rand_adv_hamming_mean = np.nanmean(rand_adv_hamming_results, axis=0)

    return [qbc_hamming_mean, rand_hamming_mean, rand_adv_hamming_mean]

# compares average hamming distance and average number of interventions for the 3 models
def performance_comparison(num_nodes: list[int], committee_size: int, epochs: int):
    df = pd.DataFrame([get_average_hamming_distances(size, 3, committee_size, epochs) for size in num_nodes])

    df.columns=[
        "qbc_hammings_means",
        "rand_hammings_means",
        "rand_adv_hammings_means",
    ]

    max_length = max(len(df[column][0]) for column in df.columns)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: pad_list(x, max_length))

    unpacked_df = pd.DataFrame({
        'qbc_hamming': df['qbc_hammings_means'].iloc[0],
        'rand_hamming': df['rand_hammings_means'].iloc[0],
        'rand_adv_hamming': df['rand_adv_hammings_means'].iloc[0]
    })

    return unpacked_df

print(performance_comparison([10], 1, 1).to_string())