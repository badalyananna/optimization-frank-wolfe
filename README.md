# Frank-Wolfe variants for cluster detection on hypergraphs

The repository implements 4 Frank-Wolfe variants to solve maximum clique problem on hypergraphs:
- Classic Frank-Wolfe (FW)
- Away-Step Frank-Wolfe (AFW)
- Pairwise Frank-Wolfe (PFW)
- Blended Pairwise Frank-Wolfe (BPFW)

## Structure of the repository
`optimizer.py` contains the main optimizer class

`utils.py` contains unitily functions, such as preprocessing the dataset, plotting

`experiments.ipynb` contains the experiments carried out in the report

`demo.ipynb` contains instuction on how to use the optimizer

`instances` folder contains the instances used for the experiments

`figures` folder contains the plots used in the report

## Dependencies
`numpy`
`scipy`
`matplotlib`
`pandas`
`seaborn`
