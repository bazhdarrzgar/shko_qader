import os
from huggingface_hub import snapshot_download

token = os.environ["HF_TOKEN"]
path = snapshot_download(
    repo_id="shkomq/Kurdish_Medical_Corpus_KMC",
    repo_type="dataset",
    token=token,
    local_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Kurdish_Medical_Corpus_KMC"),
)
print("Downloaded to:", path)
