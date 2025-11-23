"""

Creates lightweight graph-based fraud features using ZIP / CITY and AUTO_MAKE.
This helps detect suspicious clusters 
"""

import networkx as nx
import pandas as pd
import numpy as np

def build_light_graph(df: pd.DataFrame):
    """
    Builds a bipartite graph connecting:
    - insured_zip  -> auto_make
    - incident_city -> auto_make

    These patterns help reveal suspicious repeat combinations.
    """

    G = nx.Graph()

    for _, row in df.iterrows():
        make = f"MAKE_{row.get('auto_make', 'UNKNOWN')}"
        G.add_node(make, type="make")

        # Connect ZIP to auto make
        if "insured_zip" in row and pd.notna(row["insured_zip"]):
            zip_node = f"ZIP_{row['insured_zip']}"
            G.add_node(zip_node, type="zip")
            G.add_edge(zip_node, make)

        # Connect CITY to auto make
        if "incident_city" in row and pd.notna(row["incident_city"]):
            city_node = f"CITY_{row['incident_city']}"
            G.add_node(city_node, type="city")
            G.add_edge(city_node, make)

    return G


def compute_network_score(df: pd.DataFrame, G) -> pd.Series:
    """
    Computes a normalized network centrality score.
    Higher score → more connected → more suspicious.
    """

    degrees = dict(nx.degree(G))
    scores = []

    for _, row in df.iterrows():
        score = 0.0

        # ZIP contribution
        if "insured_zip" in row:
            score += degrees.get(f"ZIP_{row['insured_zip']}", 0)

        # CITY contribution
        if "incident_city" in row:
            score += degrees.get(f"CITY_{row['incident_city']}", 0)

        # AUTO MAKE contribution
        make = f"MAKE_{row.get('auto_make', 'UNKNOWN')}"
        score += degrees.get(make, 0)

        scores.append(float(score))

    scores = np.array(scores)

    # Normalize 0–1
    if scores.max() == scores.min():
        return pd.Series(np.zeros(len(scores)), index=df.index)

    norm = (scores - scores.min()) / (scores.max() - scores.min())
    return pd.Series(norm, index=df.index)
