import data_generation as dg
import pc_algorithm as pc_a
from active_learning import Experiment
import numpy as np
import pandas as pd

# remember to change committee size
def get_metrics(num_nodes: int) -> list[list]:
    G = dg.create_dag(n=num_nodes, expected_degree=3)
    pcdag = pc_a.pc(G)
    experiment = Experiment(5, 5)

    true_DAG, DAG = experiment.random_dag_from_pcdag(pcdag)

    qbc_hamming, qbc_num_interv = experiment.qbc(
        epochs=10, committee_size=5, pcdag=pcdag,
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
def get_average_metrics(num_nodes: int, num_iterations: int) -> list[int]:
    df = pd.DataFrame([get_metrics(num_nodes) for _ in range(num_iterations)])
    return df.mean(axis=0)




# compares average hamming distance and average number of interventions for the 3 models
def performance_comparison(num_nodes: list[int]):
    df = pd.DataFrame([get_average_metrics(size, 3) for size in num_nodes])

    df.columns=[
        "num_nodes",
        "qbc_hammings_means",
        "qbc_num_intervs_means",
        "rand_hammings_means",
        "rand_num_intervs_means",
        "rand_adv_hammings_means",
        "rand_adv_num_intervs_means"
    ]
    return df

performance_comparison([10])