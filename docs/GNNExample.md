# WholeGraph GNN Example

WholeMemory can be used to accelerate applications need large high bandwidth memory.
GNN applications with large graphs need large memory, both for graph structure and feature embedding vectors.
WholeGraph is one solution to accelerate large GNN training based on WholeMemory.

## Node Classification Task

Here we use [ogbn-papers100M](https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M) as an example:

WholeGraph can be used to store both graph structure and node feature embedding vectors.

### Dataset Preparation

For GNN training, we use PyTorch as the training framework, so we use multi-process chunked mode of WholeMemory.
As in multi-process mode, we use multi-process to load graph structure and embedding together.
This needs to convert the training data into WholeGraph's data format. E.g. generating the graph.
For homograph, the `examples/gnn/gnn_homograph_data_preprocess.py` can be used to convert data and build graph.
```python
WHOLEGRAPH_PATH=...
export PYTHONPATH=${WHOLEGRAPH_PATH}/python:${WHOLEGRAPH_PATH}/build:${PYTHONPATH}
python ${WHOLEGRAPH_PATH}/examples/gnn/gnn_homograph_data_preprocess.py -r ${DATASET_ROOT} -g ogbn-papers100M -p convert
python ${WHOLEGRAPH_PATH}/examples/gnn/gnn_homograph_data_preprocess.py -r ${DATASET_ROOT} -g ogbn-papers100M -p build
```
`DATASET_ROOT` is the download path of OGB datasets, defaults to `dataset`, the downloaded data will be stored here.
The converted data and built graph will be stored under `${DATASET_ROOT}/${NORMALIZED_GRAPH_NAME}/converted`

### Training

After training data is downloaded and converted. Training can be simply done by our `gnn_example_node_classification.py` script.

To run single node multi-GPU training:

```shell script
WHOLEGRAPH_PATH=...
export PYTHONPATH=${WHOLEGRAPH_PATH}/python:${WHOLEGRAPH_PATH}/build:${PYTHONPATH}
mpirun -np 8 python ${WHOLEGRAPH_PATH}/examples/gnn/gnn_example_node_classification.py -r ${DATASET_ROOT} -g ogbn-papers100M
```

This script also support multi-node training, but need to set `MASTER_ADDR` and `MASTER_PORT` correctly.

