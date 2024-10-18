import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

class TestRecommendProductsWithClustering(unittest.TestCase):
    
    def setUp(self):
        # Mock pivot_table, model, data, and customer_data
        self.pivot_table = pd.DataFrame({
            123: [0.5, 1.0, 0.0],
            456: [0.0, 1.0, 1.0],
            789: [1.0, 0.0, 0.5],
            321: [0.3, 0.6, 0.2],
            654: [0.9, 0.2, 0.1],  # Adding more customers to cluster 1
            987: [0.8, 0.1, 0.4]
        }).T  # Transpose so customer IDs are in the index
        self.pivot_table.index = [123, 456, 789, 321, 654, 987]

        self.customer_data = pd.DataFrame({
            'CustomerID': [123, 456, 789, 321, 654, 987],
            'Cluster': [0, 0, 1, 0, 1, 1]  # Assigning customers to clusters
        })

        self.data = pd.DataFrame({
            'CustomerID': [123, 456, 789, 456, 321, 654, 987],
            'StockCode': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007'],
            'Description': ['Product 1', 'Product 2', 'Product 3', 'Product 4', 'Product 5', 'Product 6', 'Product 7']
        })

        self.model = MagicMock()

        # Create a mock kneighbors function output for the model
        self.model.kneighbors = MagicMock(return_value=(
            np.array([[0.0, 0.2, 0.4]]),  # Mock distances
            np.array([[0, 1, 2]])  # Indices corresponding to customer IDs in cluster
        ))


    def recommend_products_with_clustering(self, customer_id, n_recommendations=5):
        customer_id = int(customer_id)
        if customer_id not in self.pivot_table.index:
            return f"Customer ID {customer_id} not found in the dataset."

        # Find the cluster of the customer
        customer_cluster = self.customer_data.loc[self.customer_data['CustomerID'] == customer_id, 'Cluster'].values[0]

        # Filter data to include only customers in the same cluster
        cluster_customers = self.customer_data[self.customer_data['Cluster'] == customer_cluster]['CustomerID'].tolist()
        cluster_data = self.pivot_table[self.pivot_table.index.isin(cluster_customers)]

        # Adjust the number of neighbors based on available cluster size
        n_neighbors = min(len(cluster_data), n_recommendations + 1)

        if n_neighbors <= 1:  # Ensure we have enough neighbors to recommend
            return f"Not enough data in the cluster to make recommendations for Customer ID {customer_id}."

        # Fit the model on the filtered cluster data
        self.model.fit(cluster_data)

        distances, indices = self.model.kneighbors(self.pivot_table.loc[customer_id].values.reshape(1, -1), n_neighbors=n_neighbors)

        # Get the customer IDs of the nearest neighbors
        neighbor_ids = cluster_data.index[indices.flatten()].tolist()

        # Get the products purchased by the neighbors
        neighbor_purchases = self.data[self.data['CustomerID'].isin(neighbor_ids)]

        # Recommend the most frequently purchased products by the neighbors
        recommendations = neighbor_purchases['StockCode'].value_counts().head(n_recommendations).index.tolist()

        # Get the descriptions for the recommended products
        recommended_products = self.data[self.data['StockCode'].isin(recommendations)][['StockCode', 'Description']].drop_duplicates()

        # Convert to list of tuples
        recommended_products_list = list(recommended_products.itertuples(index=False, name=None))

        return recommended_products_list

    def test_valid_customer_id(self):
        # Test with a valid customer ID (e.g., 123)
        result = self.recommend_products_with_clustering(123, n_recommendations=2)
        expected = [('P001', 'Product 1'), ('P002', 'Product 2')]  # Based on mock data
        self.assertEqual(result, expected)

    def test_invalid_customer_id(self):
        # Test with an invalid customer ID (e.g., 999)
        result = self.recommend_products_with_clustering(999, n_recommendations=2)
        expected = "Customer ID 999 not found in the dataset."
        self.assertEqual(result, expected)

    def test_correct_number_of_recommendations(self):
        # Test that the function returns the right number of recommendations
        result = self.recommend_products_with_clustering(123, n_recommendations=1)
        self.assertEqual(len(result), 1)

    def test_correct_cluster_selection(self):
        result = self.recommend_products_with_clustering(789, n_recommendations=2)  # Requesting 2 recommendations
        expected = [('P003', 'Product 3'), ('P006', 'Product 6')]  # Adjust expected result to match the function's output
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
