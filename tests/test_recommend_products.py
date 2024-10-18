import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

class TestRecommendProducts(unittest.TestCase):
    
    def setUp(self):
        # Mock pivot_table, model, and data
        self.pivot_table = pd.DataFrame({
            123: [0.5, 1.0, 0.0],
            456: [0.0, 1.0, 1.0],
            789: [1.0, 0.0, 0.5]
        }).T  # Transpose so customer IDs are in the index
        self.pivot_table.index = [123, 456, 789]
        
        self.data = pd.DataFrame({
            'CustomerID': [123, 456, 789, 456],
            'StockCode': ['P001', 'P002', 'P003', 'P004'],
            'Description': ['Product 1', 'Product 2', 'Product 3', 'Product 4']
        })
        
        self.model = MagicMock()
        
        # Create a mock kneighbors function output for the model
        self.model.kneighbors = MagicMock(return_value=(
            np.array([[0.0, 0.2, 0.4]]),  # Mock distances
            np.array([[0, 1, 2]])  # Indices corresponding to customer IDs 123, 456, 789
        ))

    def recommend_products(self, customer_id, n_recommendations=5):
        customer_id = int(customer_id)
        if customer_id not in self.pivot_table.index:
            return f"Customer ID {customer_id} not found in the dataset."

        distances, indices = self.model.kneighbors(self.pivot_table.loc[customer_id].values.reshape(1, -1), n_neighbors=n_recommendations+1)

        # Get the customer IDs of the nearest neighbors
        neighbor_ids = self.pivot_table.index[indices.flatten()].tolist()

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
        result = self.recommend_products(123, n_recommendations=2)
        expected = [('P001', 'Product 1'), ('P002', 'Product 2')]  # Based on mock data
        self.assertEqual(result, expected)

    def test_invalid_customer_id(self):
        # Test with an invalid customer ID (e.g., 999)
        result = self.recommend_products(999, n_recommendations=2)
        expected = "Customer ID 999 not found in the dataset."
        self.assertEqual(result, expected)

    def test_correct_number_of_recommendations(self):
        # Test that the function returns the right number of recommendations
        result = self.recommend_products(123, n_recommendations=1)
        self.assertEqual(len(result), 1)

if __name__ == '__main__':
    unittest.main()
