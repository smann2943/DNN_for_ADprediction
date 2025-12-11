# Introduction
The purpose of this document is to represent the differences between the results of the paper and the results from the actual run and to capture aspects of each run that could impact the differences.


## Run - 2025-11-13
Run on an NVIDIA RTX-4060 total time to completion 143,184 seconds, 2386m, 39h 77m.  The run was performed on a windows machine.

### Hyperparameter search results table
| k-fold | d.o | actual d.o | lr  | actual lr | hl size | actual hl size | #nodes | actual #nodes | acc      | actual acc |
|-------:|:---:|:----------:|:--: |:---------:|:-------:|:--------------:|:------:|:-------------:|:---:     |:----------:|
| 1      | .897| .618427    |.019 | .013      | 7       |  11            | 340    |  260          |.99857    |  .808      |
| 2      | .9  | .838       |.010 | .19       | 11      |  10            | 303    |  261          |.999571   |  .76       |
| 3      | .9  | .6         |.010 | .2        | 7       |  9             | 340    |  278          |1.0       |  .999      |
| 4      | .871| .6         |.083 | .01       | 9       |  8             | 271    |  254          |.999857   |  .995      |
| 5      | .722| .6         |.017 | .01       | 8       |  8             | 277    |  257          |.999285   |  .880      |
| avg    | .85 | .651       |.02  | .084      | 8       |  9.2           | 306    |  262          |.999710   |  .8884     |


## Run - 2025-12-08
Run on a CPU total time to completion 143,184 seconds, 2386m, 39h 77m.  The run was performed on a windows machine.

### Hyperparameter search results table
| k-fold | d.o | actual d.o | lr  | actual lr | hl size | actual hl size | #nodes | actual #nodes | acc      | actual acc |
|-------:|:---:|:----------:|:--: |:---------:|:-------:|:--------------:|:------:|:-------------:|:---:     |:----------:|
| 1      | .897| .76933     |.019 | .04       | 7       |  9             | 340    |  254          |.99857    |.8537       |
| 2      | .9  | .65348     |.010 | .06       | 11      |  10            | 303    |  300          |.99957    |.5336       |
| 3      | .9  | .72085     |.010 | .12       | 7       |  10            | 340    |  341          |1.0       |.8493       |
| 4      | .871| .82507     |.083 | .14       | 9       |  8             | 271    |  292          |.99985    |.8328       |
| 5      | .722| .63504     |.017 | .14       | 8       |  11            | 277    |  284          |.99928    |1.0         |
| avg    | .85 | .72        |.02  | .1        | 8       |  10            | 306    |  294          |.999710   |.8138       |

### ML Results
#### Random Forest
| k-fold | #genes/cpgs/samples   | Actual           | Acc /AUROC    | Actual      | AUC           | Actual AUC
|        |                       |                  | DEG + DMP     |             |
|--------|-----------------------|------------------|---------------|             |---------------|
| 1      | 295 (31,386/1946)     |                  | .659 (.652)   |             | .681 (0.670)  |  
| 2      | 236 (31,672/1840)     |                  | .745 (.739)   |             | .576 (.570)   |
| 3      | 13  (31,343/1958)     |                  | .610 (.562)   |             | .688 (.655)   |
| 4      | 19  (31,495/1890)     |                  | .763 (.750)   |             | .507 (.495)   |
| 5      | 308 (31,876/1791)     |                  | .721 (.710)   |             | .676 (.668)   |
| Average| 174.2 (31,554.4/1885) |                  | .700 (.683)   |             | .626 (.612)   |


#### SVM

####


### DNN Results
#### PCA
| k-fold | Acc      |Actual.    | Test     | Actual      | AUC           | Actual
|--------|----------|-----------|----------|             |---------------|
| 1      | 0.939    |           | 1.307    |             | .681 (0.670)  |  
| 2      | 0.856    |           | 1.760    |             | .576 (.570)   |
| 3      | 0.764    |           | 0.907    |             | .688 (.655)   |
| 4      | 0.939    |           | 1.978    |             | .507 (.495)   |
| 5      | 0.931    |           | 1.007    |             | .676 (.668)   |
| Average| 0.886    |           | 1.392    |             | .626 (.612)   |


#### TSNE

### Combined DataSets