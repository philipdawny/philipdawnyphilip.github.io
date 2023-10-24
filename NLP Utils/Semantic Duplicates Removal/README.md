


Large text corpora, especially those compiled through methods like web scraping, may often contain lines that are highly similar to each other but not exact duplicates.
These may be the exact same except maybe that a punctuation mark is missing, or one word has been replaced.

De-duplication of such text cannot be carried out through simple python operations like set function or df.drop_duplicates() as these methods look for exact matches.

The function similar_duplicates_removal can eliminate such closely similar duplicates from among a list of lines.
It employs hierarchical agglomerative clustering using sentence model embeddings.
The default sentence model used here is *all_MiniLM-L6-v2* from huggingface.

The default distance threshold for clustering has been kept as 85%, which means that lines more than 85% similar to each other will be considered as duplicates.
The distance threshold can also be altered by  passing the 'threshold' argument to the function