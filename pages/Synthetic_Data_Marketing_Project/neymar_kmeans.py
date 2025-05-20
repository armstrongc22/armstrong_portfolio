# trophy_segmentation.py

import time
import pandas as pd
import prince
import matplotlib.pyplot as plt
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.cluster import KMeans

# ── Configuration ─────────────────────────────────────────────
KEY_PATH = "mindful-vial-460001-h6-4d83b36dd3e9.json"
PROJECT  = "mindful-vial-460001-h6"
DATASET  = "euphoria"

def main(sample_limit=50000, k=4):
    start_all = time.perf_counter()

    # 1) Authenticate & init BigQuery client
    creds  = service_account.Credentials.from_service_account_file(KEY_PATH)
    client = bigquery.Client(project=PROJECT, credentials=creds)

    # 2) Query trophy buyer profiles (limited sampling)
    print("Querying trophy buyer profiles...")
    sql = f"""
    WITH trophy_profiles AS (
      SELECT
        c.customer_id,
        DATE_DIFF(CURRENT_DATE(), DATE(c.birthday), YEAR) AS age,
        c.gender,
        c.region
      FROM `{PROJECT}.{DATASET}.purchase_events_topic` p
      JOIN `{PROJECT}.{DATASET}.customers_topic`    c
        USING(customer_id)
      WHERE p.category = 'merch'
        AND p.product_name = 'Authentic Mahiman Trophy'
    )
    SELECT *
    FROM trophy_profiles
    LIMIT {sample_limit}
    """
    df = client.query(sql).to_dataframe()
    print(f"  → Loaded {len(df)} rows in {time.perf_counter() - start_all:.1f}s")

    # 3) Bin age and prepare for MCA
    df['age_bin'] = pd.cut(
        df['age'],
        bins=range(10, 81, 5),
        labels=[f"{i}-{i+4}" for i in range(10, 80, 5)],
        right=False
    )
    df_mca = df[['age_bin', 'gender', 'region']].astype(str)

    # 4) Run MCA
    print("Performing MCA...")
    mca = prince.MCA(n_components=2, engine='sklearn', random_state=42)
    mca = mca.fit(df_mca)
    coords = mca.transform(df_mca)
    coords.columns = ['Dim1', 'Dim2']

    # 5) Compute inertia
    eigen = mca.eigenvalues_
    inertia = eigen / eigen.sum()
    print(f"  Dim1 inertia: {inertia[0]:.2%}, Dim2 inertia: {inertia[1]:.2%}")

    # 6) K-Means on MCA dims
    print(f"Clustering into k={k} segments...")
    km = KMeans(n_clusters=k, random_state=42)
    coords['cluster'] = km.fit_predict(coords[['Dim1', 'Dim2']])
    df['cluster'] = coords['cluster']

    # 7) Output cluster profiles
    print("\nCluster summaries:")
    for i in range(k):
        seg = df[df.cluster == i]
        print(f"\nCluster {i}:")
        print(f"  Size: {len(seg)}")
        print("  Top regions:", seg.region.value_counts().head(5).to_dict())
        print(f"  Avg age: {seg.age.mean():.1f}")

    # 8) Plot result
    plt.figure(figsize=(8, 6))
    plt.scatter(
        coords['Dim1'], coords['Dim2'],
        c=coords['cluster'], cmap='tab10', alpha=0.6
    )
    centers = km.cluster_centers_
    plt.scatter(
        centers[:,0], centers[:,1],
        marker='X', s=200, c='black'
    )
    plt.axhline(0, color='gray', lw=0.8)
    plt.axvline(0, color='gray', lw=0.8)
    plt.xlabel('Dim1')
    plt.ylabel('Dim2')
    plt.title('MCA + KMeans Segments')
    plt.show()

if __name__ == "__main__":
    main()
