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
