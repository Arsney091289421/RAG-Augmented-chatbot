-- Query the number of chunks per document
SELECT 
  source_file,
  COUNT(*) AS chunk_count
FROM 
  `your_project.dataset_for_rag_embeddings.transformers_embeddings`
GROUP BY 
  source_file
ORDER BY 
  chunk_count DESC;

-- Query the average chunk length per document
SELECT
  source_file,
  AVG(LENGTH(content)) AS avg_chunk_length
FROM 
  `your_project.dataset_for_rag_embeddings.transformers_embeddings`
GROUP BY 
  source_file
ORDER BY 
  avg_chunk_length DESC;

-- Query chunks containing the keyword "Transformer"
SELECT 
  source_file,
  chunk_id,
  content
FROM 
  `your_project.dataset_for_rag_embeddings.transformers_embeddings`
WHERE 
  LOWER(content) LIKE '%transformer%'
LIMIT 10;
