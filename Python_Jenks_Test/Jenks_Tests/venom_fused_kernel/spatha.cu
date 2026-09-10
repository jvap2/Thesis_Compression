/*
 * Copyright (C) 2023 Roberto Lopez Castro (roberto.lopez.castro@udc.es). All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../spmm/spmm_op.h"
#include "../spmm/spmm_library_decl.h"
#include "../spmm/blockwise_library.cu"

#include <torch/extension.h>
namespace py = pybind11;

using namespace spatha;

/* torch::Tensor spmm_cuda(torch::Tensor A_metadata,
                        torch::Tensor A_indices,
                        torch::Tensor A_values,
                        torch::Tensor rhs_matrix,
                        torch::Tensor output_matrix,
                        int A_num_rows,
                        int A_num_cols,
                        int B_num_cols,
                        int vec_length,
                        int n,
                        int m,
                        int nnz,
                        int seed,
                        int mbrow,
                        int brow); */

torch::Tensor spmm(torch::Tensor A_metadata,
                        torch::Tensor A_indices,
                        torch::Tensor A_values,
                        torch::Tensor rhs_matrix,
                        torch::Tensor bias,
                        int A_num_rows,
                        int A_num_cols,
                        int B_num_cols,
                        int vec_length,
                        int n,
                        int m,
                        int nnz,
                        int seed,
                        int mbrow,
                        int brow,
                        int L_conv, int cOW=0, int cC=0, int cKW=0, int cKHKW=0, int cHp=0, int cWp=0, torch::Tensor col_off=torch::Tensor())
{
    return spmm_cuda(A_metadata, A_indices, A_values, rhs_matrix, bias,
                        A_num_rows, A_num_cols, B_num_cols,
                        vec_length, n, m, nnz,
                        seed, mbrow, brow, L_conv, cOW, cC, cKW, cKHKW, cHp, cWp,
                        col_off.numel()>0 ? col_off.data_ptr<int>() : nullptr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("spmm",  &spmm,  "Custom SPMM kernel", py::arg("A_metadata"),py::arg("A_indices"),py::arg("A_values"),py::arg("rhs_matrix"),py::arg("bias"),py::arg("A_num_rows"),py::arg("A_num_cols"),py::arg("B_num_cols"),py::arg("vec_length"),py::arg("n"),py::arg("m"),py::arg("nnz"),py::arg("seed"),py::arg("mbrow"),py::arg("brow"),py::arg("L_conv")=0,py::arg("cOW")=0,py::arg("cC")=0,py::arg("cKW")=0,py::arg("cKHKW")=0,py::arg("cHp")=0,py::arg("cWp")=0,py::arg("col_off")=torch::Tensor());
}