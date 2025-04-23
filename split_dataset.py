import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def kennard_stone(X, n_samples):
    """
    Simple Kennard-Stone algorithm for sample selection.
    X: numpy array of shape (n_samples_total, n_features)
    n_samples: number of samples to select
    Returns: list of selected indices
    """
    X = np.asarray(X)
    n_total = X.shape[0]
    selected = []
    remaining = list(range(n_total))

    # Start with the two most distant points
    dist_matrix = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    selected.extend([i, j])
    remaining.remove(i)
    remaining.remove(j)

    while len(selected) < n_samples:
        dists = np.min(dist_matrix[remaining][:, selected], axis=1)
        next_idx = remaining[np.argmax(dists)]
        selected.append(next_idx)
        remaining.remove(next_idx)

    return selected

def main(input_csv, test_csv, val_csv, train_csv, test_frac=0.15, val_frac=0.15, n_clusters=3):
    # For reproducibility
    np.random.seed(42)

    # Read CSV; ignore the first column "aurl" if it exists
    df = pd.read_csv(input_csv)
    if df.columns[0] == 'aurl':
        df = df.drop(df.columns[0], axis=1)
    df = df.reset_index(drop=True)

    # Error handling for missing egap values
    if 'egap' not in df.columns:
        raise ValueError("'egap' column not found in the input CSV.")
    if df['egap'].isnull().any():
        raise ValueError("Missing values found in 'egap' column.")
    
    # Check for spacegroup_relax column
    if 'spacegroup_relax' not in df.columns:
        raise ValueError("'spacegroup_relax' column not found in the input CSV.")
    
    # Ensure spacegroup_relax is treated as categorical
    df['spacegroup_relax'] = df['spacegroup_relax'].astype('category')
    
    # Separate numerical and categorical features
    cat_features = ['spacegroup_relax']
    num_features = [col for col in df.columns if col not in cat_features]
    
    # Create preprocessing pipeline that handles both numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_features)
        ])
    
    # Apply preprocessing to get scaled features for Kennard-Stone
    X_processed = preprocessor.fit_transform(df)
    
    n_total = df.shape[0]
    n_test = int(np.round(test_frac * n_total))
    n_val = int(np.round(val_frac * n_total))

    # Before selecting test set, group by spacegroup and ensure representation
    spacegroups = df['spacegroup_relax'].unique()
    test_indices = []
    remaining_indices = set(range(len(df)))

    # For each spacegroup, select proportional samples for test set
    for sg in spacegroups:
        sg_indices = df[df['spacegroup_relax'] == sg].index.tolist()
        if len(sg_indices) > 0:
            # Ensure at least 60% of samples for each spacegroup stay in training
            max_test_samples = int(len(sg_indices) * min(test_frac, 0.2))
            if max_test_samples > 0:
                # Use Kennard-Stone on this spacegroup's samples
                sg_data = X_processed[sg_indices]
                sg_test_idx = kennard_stone(sg_data, max_test_samples)
                test_indices.extend([sg_indices[i] for i in sg_test_idx])
                remaining_indices -= set([sg_indices[i] for i in sg_test_idx])

    # Convert remaining indices to list and create test set
    remaining_indices = list(remaining_indices)
    test_set = df.iloc[test_indices].reset_index(drop=True)
    df_rem = df.iloc[remaining_indices].reset_index(drop=True)

    # Process remaining data for clustering
    X_rem_processed = preprocessor.transform(df_rem)
    
    # Apply KMeans clustering on the preprocessed features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_rem_processed)
    df_rem['cluster'] = clusters

    val_list = []
    train_list = []
    
    # Modified validation split to ensure representation
    for cl in np.unique(clusters):
        cluster_df = df_rem[df_rem['cluster'] == cl].copy()
        
        # Group by spacegroup within cluster
        for sg in cluster_df['spacegroup_relax'].unique():
            sg_cluster_df = cluster_df[cluster_df['spacegroup_relax'] == sg]
            
            if len(sg_cluster_df) < 4:
                # Add small groups directly to training
                train_list.append(sg_cluster_df)
                continue
                
            # Calculate validation size while ensuring at least 50% remains for training
            sg_val_size = min(val_frac, 0.3)  # Changed val_ratio to val_frac
            
            try:
                # Split while preserving spacegroup distribution
                train_sg, val_sg = train_test_split(
                    sg_cluster_df,
                    test_size=sg_val_size,
                    stratify=sg_cluster_df['spacegroup_relax'],
                    random_state=42
                )
                train_list.append(train_sg)
                val_list.append(val_sg)
            except ValueError:
                # If stratification fails, add all to training
                train_list.append(sg_cluster_df)

    train_set = pd.concat(train_list, ignore_index=True)
    val_set = pd.concat(val_list, ignore_index=True)

    # Verify spacegroup coverage
    train_groups = set(train_set['spacegroup_relax'].unique())
    test_groups = set(test_set['spacegroup_relax'].unique())
    val_groups = set(val_set['spacegroup_relax'].unique())
    
    missing_in_train = (test_groups | val_groups) - train_groups
    if missing_in_train:
        print("\nWARNING: Following spacegroups in test/val are missing from training:")
        print(missing_in_train)
        
        # Move some samples from test/val to train for missing groups
        for sg in missing_in_train:
            # Collect samples from test and val
            sg_samples = pd.concat([
                test_set[test_set['spacegroup_relax'] == sg],
                val_set[val_set['spacegroup_relax'] == sg]
            ])
            
            if not sg_samples.empty:
                # Keep one sample for training
                train_list.append(sg_samples.iloc[[0]])
                
                # Update test and val sets
                test_set = test_set[test_set['spacegroup_relax'] != sg]
                val_set = val_set[val_set['spacegroup_relax'] != sg]

    # Write out the test, validation, and training sets to separate files
    test_set.to_csv(test_csv, index=False)
    val_set.to_csv(val_csv, index=False)
    train_set.to_csv(train_csv, index=False)

    actual_val_count = len(val_set)
    print("\nSplitting complete!")
    print("Test set shape:", test_set.shape)
    print(f"Target val samples: {n_val}, Actual: {actual_val_count}")
    print("Validation set shape:", val_set.shape)
    print("Training set shape:", train_set.shape)
    
    # Print distribution of spacegroups across splits for verification
    print("\nSpacegroup distribution in test set:")
    print(test_set['spacegroup_relax'].value_counts().head(5))
    print("\nSpacegroup distribution in validation set:")
    print(val_set['spacegroup_relax'].value_counts().head(5))
    print("\nSpacegroup distribution in training set:")
    print(train_set['spacegroup_relax'].value_counts().head(5))
    
    # Print comprehensive spacegroup distribution analysis
    print("\nComplete Spacegroup Distribution Analysis:")
    print("-" * 50)
    
    print("\nOriginal Dataset:")
    orig_dist = df['spacegroup_relax'].value_counts()
    print(orig_dist)
    
    print("\nTest Set Distribution:")
    test_dist = test_set['spacegroup_relax'].value_counts()
    print(test_dist)
    
    print("\nValidation Set Distribution:")
    val_dist = val_set['spacegroup_relax'].value_counts()
    print(val_dist)
    
    print("\nTraining Set Distribution:")
    train_dist = train_set['spacegroup_relax'].value_counts()
    print(train_dist)
    
    # Calculate percentages
    print("\nPercentage Distribution in Each Set:")
    print("-" * 50)
    print(f"{'Spacegroup':<10} {'Original%':>10} {'Test%':>10} {'Val%':>10} {'Train%':>10}")
    print("-" * 50)
    
    all_spacegroups = sorted(df['spacegroup_relax'].value_counts().index)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Split CSV dataset using modAL Kennard–Stone, feature scaling, clustering-based stratification, and spacegroup handling.'
    )
    parser.add_argument('input_file', help='Path to input CSV file.')  # Changed from raw_data.csv
    parser.add_argument('test_file', help='Output CSV file for the test set.')  # Changed from test.csv
    parser.add_argument('val_file', help='Output CSV file for the validation set.')  # Changed from val.csv 
    parser.add_argument('train_file', help='Output CSV file for the training set.')  # Changed from train.csv
    parser.add_argument('--test_frac', type=float, default=0.15, help='Fraction of data to use as test set.')
    parser.add_argument('--val_frac', type=float, default=0.15, help='Fraction of original data to use as validation set.')
    parser.add_argument('--n_clusters', type=int, default=3, help='Number of clusters for clustering-based split.')
    args = parser.parse_args()

    main(args.input_file, args.test_file, args.val_file, args.train_file, 
         args.test_frac, args.val_frac, args.n_clusters)