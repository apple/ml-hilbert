#  Hilbert: Recursively Building Formal Proofs with Informal Reasoning

This software project accompanies the research paper, [Hilbert: Recursively Building Formal Proofs with Informal Reasoning](https://arxiv.org/pdf/2509.22819).


## Getting Started

### Prerequisites

In order to setup Hilbert, you will need to install some packages. Run the following commands

#### Step 1: Create a `conda` environment with the necessary dependencies.
```
conda create -n hilbert python=3.10 -y
conda activate hilbert
# Install `kimina-lean-server`
git clone https://github.com/project-numina/kimina-lean-server.git
cd kimina-lean-server
cp .env.template .env # Optional
bash setup.sh # Installs Lean, repl and mathlib4
pip install -r requirements.txt
pip install .
python3 -m prisma generate
cd ..
```
#### Step 2: Install some dependencies
```
# Install dependencies
pip install -U sentence_transformers
pip install hydra-core omegaconf httpx openai numpy faiss-cpu tqdm pydantic pygments \
           typing_extensions tenacity nest_asyncio axlearn
```
#### Step 3: Setup the `cache` folder
```
# Download `mathlib_informal`
mkdir cache && cd cache
wget https://huggingface.co/datasets/FrenzyMath/mathlib_informal_v4.16.0/resolve/main/data.jsonl
mv data.jsonl mathlib_informal.jsonl

# Save mpnet-base-v2 to cache
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2
cd all-mpnet-base-v2
git lfs pull
cd ../../
```

#### Step 4 (optional): Create a sample dataset
```
mkdir data && cd data
# Create a sample dataset
mkdir sample_dataset && cd sample_dataset
echo '{"name": "sqrt_10_irrational", "header": "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n", "formal_statement": "theorem sqrt_ten_irrational : Irrational (Real.sqrt (10 : ℝ)) := by\n", "split": "test", "informal_prefix": "/--prove that sqrt(10) is irrational--/"}' > sample.jsonl
```


### Running the experiments

Before running the experiments, you will need to setup OpenAI-compatible end-points to the prover LLM and the reasoner LLM. You can either use remote end-points or locally host LLMs (for example, with [vLLM](https://github.com/vllm-project/vllm)).

To run the experiments, first setup `kimina-lean-server` in a separate terminal.
```
cd kimina-lean-server
source "$HOME/.elan/env"
LEAN_SERVER_PORT=10001 python -m server
```

Then, change the URLs in the config files in `configs/`. In particular, make sure the `base_url` attributes for the `informal_llm` and `prover_llm` are setup correctly.

In a new terminal, run
```
OPENAI_API_KEY=<insert api key here> python -m src.run data=sample_dataset
```
If you are running local models, you might want to insert a placeholder key (e.g. `'empty'`) as the API key.

#### Running with other datasets

In order to run the experiments with other datasets like MiniF2F or PutnamBench, you will have to store the data as JSONL files in the `data/` directory and edit the config files in `configs/data`.

For MiniF2F, we recommend using a JSONL version (for example, like [this one](https://huggingface.co/datasets/Tonic/MiniF2F/raw/main/minif2f.jsonl)).

For PutnamBench, you can use [this script](https://github.com/trishullab/PutnamBench/blob/main/lean4/scripts/extract_to_json.py) to convert the PutnamBench dataset to JSONL format.

### Contributing

Thanks for your interest in contributing. This project was released to accompany a research paper for purposes of reproducibility, and beyond its publication there are limited plans for future development of the repository.

While we welcome new pull requests and issues please note that our response may be limited. Forks and out-of-tree improvements are strongly encouraged.

### Citation

If you found our work useful, consider citing our paper.

```
@article{varambally2025hilbert,
  title={Hilbert: Recursively Building Formal Proofs with Informal Reasoning},
  author={Varambally, Sumanth and Voice, Thomas and Sun, Yanchao and Chen, Zhifeng and Yu, Rose and Ye, Ke},
  journal={arXiv preprint arXiv:2509.22819},
  year={2025}
}
```