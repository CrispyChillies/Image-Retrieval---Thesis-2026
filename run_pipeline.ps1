param(
    [string]$ModelType = "densenet121",
    [string]$Dataset = "chestxray",
    [string]$DataDir,
    [string]$ImageList,
    [string]$ModelWeights,
    [switch]$UseZilliz,
    [string]$Uri,
    [string]$Token,
    [string]$SaveNp = "./embeddings_pipeline.npz",
    [int]$Limit = 5000,
    [int]$ClusterK = 3,
    [switch]$Interactive,
    [int]$UmapNeighbors = 15,
    [float]$UmapMinDist = 0.1
)

if (-not $DataDir -or -not $ImageList -or -not $ModelWeights) {
    Write-Error "Provide -DataDir, -ImageList, and -ModelWeights"
    exit 1
}

# 1) Setup collection
Write-Host "Setting up Milvus collection for $ModelType / $Dataset"
if ($UseZilliz) {
    python milvus/milvus_setup.py --model $ModelType --dataset $Dataset --uri $Uri --token $Token
} else {
    python milvus/milvus_setup.py --model $ModelType --dataset $Dataset
}

# 2) Ingest and save embeddings
Write-Host "Ingesting embeddings (and saving to $SaveNp)"
$ingestCmd = "python ingest_embeddings.py --model_type $ModelType --model_weights \"$ModelWeights\" --data_dir \"$DataDir\" --image_list \"$ImageList\" --embedding_dim 1024 --device cuda --batch_size 32 --insert_batch_size 100 --store-local-paths --dataset $Dataset --save_embeddings_npz \"$SaveNp\""
if ($UseZilliz) {
    $ingestCmd += " --uri $Uri --token $Token"
}
Write-Host $ingestCmd
Invoke-Expression $ingestCmd

# 3) Visualize
Write-Host "Visualizing embeddings"
$vizCmd = "python visualize_embeddings.py --embeddings_file \"$SaveNp\" --method umap --limit $Limit --out embeddings_viz.png --points_csv embeddings_points.csv --marker_size 10 --alpha 0.7 --dpi 300 --cluster_k $ClusterK --umap_n_neighbors $UmapNeighbors --umap_min_dist $UmapMinDist"
if ($Interactive) { $vizCmd += " --interactive" }
Write-Host $vizCmd
Invoke-Expression $vizCmd

Write-Host "Pipeline complete. Output: embeddings_viz.png, embeddings_points.csv"