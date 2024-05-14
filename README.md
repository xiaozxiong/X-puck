# Usage

### Build

```shell
source /opt/intel/oneapi/mkl/latest/env/vars.sh
cmake -DCMAKE_BUILD_TYPE=Release -DMKLROOT=${MKLROOT} -DBLA_VENDOR=Intel10_64lp_seq -DBLA_STATIC=ON -B build .
cd build && make -j
```

### Train and Build
Firstly, we need to create directory `output` to store index and `dataset` to link dataset directory. The input file is in `.fvecs` format.

```shell
ln -s /mnt/data1/xzx/tinker_output output
ln -s /mnt/data0/ANN-Datasets dataset
```
The `.conf` file in `tools/demo/conf/` can modified to set traing and building parameter:
- `tools/demo/conf/puck_train.conf`
- `tools/demo/conf/tinker_train.conf`

Then we can run the shell script `puck_train_control.sh` in `tools/script/` to train and build index. It is worthy to note that the current path of `puck_train_control.sh` is `exe_puck/` and the parameters in `.conf` file shoule be set based on it. If this script fails to run, the possible reason is the writing permission and you should add `sudo`.

```shell
cd tools/script
./puck_train_control.sh -t -b
```
After that, all index information would be stored in derectory `output` for specified dataset such as `output/sift10K`.

### Search

We can set search parameters in:
- `demo/conf/puck.conf`
- `demo/conf/tinker.conf`

Then we can run the following command to perform searching, the query file is in `.fvecs` format and gt file is in `.ivecs` format.
```shell
./bin/search_client QUERY_FILE GT_FILE --flagfile=demo/conf/puck.conf
# or
./bin/search_client QUERY_FILE GT_FILE --flagfile=demo/conf/tinker.conf
# 
```
For example:
```shell
./bin/search_client dataset/deep1M/deep1M_query.fvecs dataset/deep1M/deep1M_groundtruth.ivecs --flagfile=demo/conf/tinker.conf
```
