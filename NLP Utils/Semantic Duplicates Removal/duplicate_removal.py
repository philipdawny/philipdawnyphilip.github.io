# Install dependencies
#!pip install sentence-transformers

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformers, Util

# Initialise language model
model = SentenceTransformers('all_MiniLM-L6-v2')

class duplicate_removal:


    def similar_duplicates_removal(self, list_text, threshold = 0.85):

        # Distance threshold for clustering is 1 - percentage similarity
        distance_threshold = 1-threshold


        # Create embeddings for the input list using the sentence model
        corpus = list_text
        corpus_embeddings = model.encode(corpus)
        corpus_embeddings = corpus_embeddings/np.linalg.norm(corpus_embeddings, axis = 1, keepdims = True)


        # Initialize agglomerative clutering model with distance threshold and cosine distance criteria
        clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold = distance_threshold,
                                                   affinity = 'cosine', linkage = 'average')

        # Fit the clustering model on input data
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        # Group lines that appear in the same cluster
        clustered_sentences = {}
        for sent_no, cluster_no in enumerate(cluster_assignment):
            if cluster_no not in clustered_sentences:
                clustered_sentences[cluster_no] = []

            clustered_sentences[cluster_no].append(corpus[sent_no])

        # Create a Dataframe of the clusters
        clusters_df = pd.DataFrame(list(clustered_sentences.items()), columns = ['cluster_no', 'cluster'])



        centroids = []

        # For each cluster, calculate the centroid vector as mean of vectors of its elements
        for i in np.unique(cluster_assignment):
            centroid = np.mean(corpus_embeddings[cluster_assignment == i], axis = 0)
            centroids.append(centroid)

        # Find representative element for each cluster

        representative_lines = []

        # For each cluster, the representative element is the cluster element closest (cosine) to the corresponding previously calculated centroid vectors
        for i, tuple in enumerate(centroids):
            id = tuple[0]
            centroid= tuple[1]
            distances = cosine_similarity(corpus_embeddings, [centroid])
            nearest_document_index = np.argmax(distances[:, 0])
            nearest_document = corpus[nearest_document_index]
            clusters_df.loc[clusters_df.cluster_no == id, 'centroid'] = nearest_document
            representative_lines.append(nearest_document)


        # Return the list of cluster centroids
        return clusters_df.centroid.to_list()
