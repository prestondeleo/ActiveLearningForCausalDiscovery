import data_generation as dg
import pc_algorithm as pc_a
from active_learning import Experiment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# remember to change committee size
def get_metrics(num_nodes: int, committee_size: int, epochs: int):
    G = dg.create_dag(n=num_nodes, expected_degree=3)
    pcdag = pc_a.pc(G)
    experiment = Experiment(5, 5)

    true_DAG, DAG = experiment.random_dag_from_pcdag(pcdag)

    start_time = time.time()
    qbc_hamming, qbc_num_interv = experiment.qbc(
        epochs=epochs, committee_size=committee_size, pcdag=pcdag,
        true_causal_dag=true_DAG, true_causal_graph=DAG,
        data=dg.generate_data(DAG), k=num_nodes, _lambda=0.5
    )
    qbc_runtime = time.time() - start_time
    print(f"qbc runtime: {qbc_runtime:.4f} seconds")

    start_time = time.time()
    rand_hamming, rand_num_interv = experiment.random_design(
        pcdag=pcdag, true_causal_graph=DAG,
        true_causal_dag=true_DAG, data=dg.generate_data(DAG), k=num_nodes
    )
    rand_runtime = time.time() - start_time

    start_time = time.time()
    rand_adv_hamming, rand_adv_num_interv = experiment.random_adv_design(
        pcdag=pcdag, true_causal_graph=DAG,
        true_causal_dag=true_DAG, data=dg.generate_data(DAG), k=num_nodes
    )
    rand_adv_runtime = time.time() - start_time

    return [qbc_hamming, rand_hamming, rand_adv_hamming], qbc_runtime, rand_runtime, rand_adv_runtime

def pad_list(lst, length):
    if isinstance(lst, np.ndarray):
        lst = lst.tolist()

    if len(lst) == 0:
        return [0.0] * length

    return lst + [0.0] * (length - len(lst))

def get_average_hamming_distances(num_nodes: int, num_iterations: int, committee_size: int, epochs: int):
    qbc_hamming_results = []
    rand_hamming_results = []
    rand_adv_hamming_results = []

    qbc_runtimes = []
    rand_runtimes = []
    rand_adv_runtimes = []

    for i in range(num_iterations):
        print(f"Iteration{i+1}")

        qbc_hamming, rand_hamming, rand_adv_hamming = get_metrics(num_nodes, committee_size, epochs)[0]
        qbc_hamming_results.append(qbc_hamming)
        rand_hamming_results.append(rand_hamming)
        rand_adv_hamming_results.append(rand_adv_hamming)

        qbc_runtimes.append(get_metrics(num_nodes, committee_size, epochs)[1])
        rand_runtimes.append(get_metrics(num_nodes, committee_size, epochs)[2])
        rand_adv_runtimes.append(get_metrics(num_nodes, committee_size, epochs)[3])

    qbc_pad = max(len(ls) for ls in qbc_hamming_results)
    rand_pad = max(len(ls) for ls in rand_hamming_results)
    rand_adv_pad = max(len(ls) for ls in rand_adv_hamming_results)

    qbc_hamming_results = [pad_list(lst, qbc_pad) for lst in qbc_hamming_results]
    rand_hamming_results = [pad_list(lst, rand_pad) for lst in rand_hamming_results]
    rand_adv_hamming_results = [pad_list(lst, rand_adv_pad) for lst in rand_adv_hamming_results]

    qbc_hamming_results = np.array(qbc_hamming_results)
    rand_hamming_results = np.array(rand_hamming_results)
    rand_adv_hamming_results = np.array(rand_adv_hamming_results)

    qbc_hamming_mean = np.nanmean(qbc_hamming_results, axis=0)
    rand_hamming_mean = np.nanmean(rand_hamming_results, axis=0)
    rand_adv_hamming_mean = np.nanmean(rand_adv_hamming_results, axis=0)

    return [[qbc_hamming_mean, rand_hamming_mean, rand_adv_hamming_mean], np.mean(qbc_runtimes),
            np.mean(rand_runtimes), np.mean(rand_adv_runtimes)]

# compares average hamming distance and average number of interventions for the 3 models
def performance_comparison(num_nodes: list[int], num_iterations: int, committee_size: int, epochs: int):
    #df1 = pd.DataFrame([get_average_hamming_distances(size, num_iterations, committee_size, epochs)[0] for size in
     #                  num_nodes])

    results=[]
    qbc_times = []
    rand_times = []
    rand_adv_times = []

    for size in num_nodes:
        results.append(get_average_hamming_distances(size, num_iterations, committee_size, epochs)[0])
        qbc_times.append(get_average_hamming_distances(size, num_iterations, committee_size, epochs)[1])
        rand_times.append(get_average_hamming_distances(size, num_iterations, committee_size, epochs)[2])
        rand_adv_times.append(get_average_hamming_distances(size, num_iterations, committee_size, epochs)[3])

    df1 = pd.DataFrame(results)

    df1.columns=[
        "qbc_hammings_means",
        "rand_hammings_means",
        "rand_adv_hammings_means",
    ]

    df2 = pd.DataFrame({
        "qbc average time": qbc_times,
        "rand average time": rand_times,
        "rand_adv average time": rand_adv_times
    })

    max_length = max(len(df1[column][0]) for column in df1.columns)
    for col in df1.columns:
        df1[col] = df1[col].apply(lambda x: pad_list(x, max_length))

    unpacked_df = pd.DataFrame({
        'qbc': df1['qbc_hammings_means'].iloc[0],
        'rand': df1['rand_hammings_means'].iloc[0],
        'rand_adv': df1['rand_adv_hammings_means'].iloc[0]
    })

    print(df2)
    return unpacked_df, df2

def save_performance_info(num_nodes: list[int], num_iterations: int, committee_size: int, epochs: int):
    df1 = performance_comparison(num_nodes, num_iterations, committee_size, epochs)[0]
    df1.to_csv(f'performance_comp_{num_nodes}_nodes_{num_iterations}_iterations_{committee_size}_committee_'
              f'{epochs}_epochs')

    df2 = performance_comparison(num_nodes, num_iterations, committee_size, epochs)[1]
    df2.to_csv(f'time_comp_{num_nodes}_nodes_{num_iterations}_iterations_{committee_size}_committee_'
               f'{epochs}_epochs')

def plot(num_nodes: list[int], num_iterations: int, committee_size: int, epochs: int):
    df = performance_comparison(num_nodes, num_iterations, committee_size, epochs)[0]

    plt.figure(figsize=(10, 6))

    for column in df.columns:
        plt.plot(df.index, df[column], label=column)

    plt.title('Averge Hamming Distance by Number of Interventions')
    plt.xlabel('# Interventions')
    plt.ylabel('Average Hammming Distance')
    plt.legend()

    plt.show()

plot([10], 3, 1, 1)