

## These are some of the NLP utility functions that I regurarly use for NLP tasks


The class 'nlp_utils' can be imported into any python program and the functions contained in it can be called with the respective parameters.


`text_clustering()`

This function carries out text clustering based on sentence embeddings.
The default model selected is 'gte-large' from [https://huggingface.co/thenlper/gte-large](url)

Default threshold set for clustering is 01. which means that corpus elements at a cosine distance greater than 0.1 will be placed in separate clusters.

Keep in mind that this threshold is applicable only for the model 'gte-large'. In case user selects another model for clustering, the threshold will have to be adjusted accordingly.

Cluster centroid is calculated as the cluster element that is closest (cosine) to the mean of all cluster element vectors.

The default sentence model can be overriden by passing in the 'model_name' parameter, and the threshold can be adjusted using the 'input_thres' parameter.

This functions returns a Pandas DataFrame with cluster IDs, cnetroid element, cluster size and cluster elements.







`similar_duplicates_removal()`

Large text corpora, especially those compiled through methods like web scraping, may often contain lines that are highly similar but not exact duplicates of each other.These may be exactly identical except maybe that a punctuation mark is missing, or one word has been replaced.

De-duplication of such text cannot be carried out through simple python operations like set function or df.drop_duplicates() as these methods look for exact matches.


The function similar_duplicates_removal can eliminate such closely similar duplicates from among a list of lines.
It employs hierarchical agglomerative clustering using sentence model embeddings.
The default sentence model used here is all_MiniLM-L6-v2 from huggingface.

The default distance threshold for clustering has been kept as 85%, which means that lines more than 85% similar to each other will be considered as duplicates.
The distance threshold can also be altered by  passing the 'threshold' argument to the function.

The function returns the list of unique lines.



