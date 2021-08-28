
#### How To Compile

```bash
cd simple_net
mkdir build && cmake ..
make -j
```

#### How To Run

```bash
./build/simple_net data/digits_1.csv data/weights.csv
```

##### Results

###### Part 1 `ops::DenseLayer().SimpleForword`

```bash
Time elapsed for Part 1: 0.031030s
The prediction is: 96.661102% Pass:1737 Total:1797
```

###### Part 2 `ops::DenseLayer().OptForword`

```bash
Time elapsed for Part 2: 0.033443s
The prediction is: 96.661102% Pass:1737 Total:1797
```

1. Vectoriza reduce axis `k=[0,64)`.
    ```c++
    // TODO: avx2
    for (int32_t ko = 0; ko < in_dim / factor; ++ko) {
      for (int32_t k_inner = 0; k_inner < factor; ++k_inner) {
        dense_out[i * out_dim + j] += data[i * in_dim + ko * factor + k_inner] * weight[j * in_dim + ko * factor + k_inner];
      }
    }
    ```

2. Parallel axis `j=[0,10)`.
    ```c++
    for (int32_t j = 0; j < out_dim; ++j) {
       // ... 
    }
    ```

3. Block axis `i=[0, 1000)`.
    ```c++
    for (int32_t i = 0; j < batch; ++i) {
      for (int32_t j = 0; j < out_dim; ++j) {
         // ... 
      }
    }
    ```

4. Reorder axis.
   ```c++
   TODO
   ```

5. Packing.
   ```c++
   TODO
   ```
