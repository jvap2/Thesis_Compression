
        #include <iostream>
        #include <algorithm>
        #include <utility>
        #include <cstdlib>
        #include <cstdio>
        #include <cmath>
        #include <functional>
        #include <tuple>
        #include <vector>
        #include <numeric>
        #include <chrono>

        using namespace std;


        extern "C" void func2(float* sparse, int* masks, float* hA_values, int *hA_columns, int *hA_metadata){

            int bm_m = 64/64;
            int mbrow_m = 64/32;
            int mbrow_m2 = 32/16;
            int brow_m = 16/4;
            // metadata
            int mcol_kk = 16/2/2;
            int mcol_k = 80/2/mcol_kk;
            // indices
            int col_kk = mcol_kk;
            int col_k = 80/2/col_kk;

            float values[16];
            uint indexes[16];
            uint columns[col_kk*4];

            int max_idx = 0;

            for(int bm_i=0; bm_i<bm_m; bm_i++){
                for(int mbrow_i=0; mbrow_i<mbrow_m; mbrow_i++){
                    for(int mbrow_i2=0; mbrow_i2<mbrow_m2; mbrow_i2++){
                        for(int brow_i=0; brow_i<brow_m; brow_i++){
                            for(int mcol_i=0; mcol_i<mcol_k; mcol_i++){
                                for(int col_i=0; col_i<col_kk; col_i++){
                                    for(int col_ii=0; col_ii<4; col_ii++){
                                        columns[col_i*4 + col_ii] =
                                        hA_columns[bm_i*col_k*col_kk*4 + mcol_i*col_kk*4 + col_i*4 + col_ii];
                                    }
                                }
                                for(int mbrow_ii=0; mbrow_ii<(4/2); mbrow_ii++){
                                    for(int mcol_ii=0; mcol_ii<mcol_kk; mcol_ii++){
                                        for(int mbrow_iii=0; mbrow_iii<2; mbrow_iii++){
                                            int pos=0;
                                            for(int n_i=0; n_i<4; n_i++){
                                                unsigned int index = columns[mcol_ii*4 + n_i];

                                                if((mcol_i*16*mcol_kk + mcol_ii*16 + index) < 576){
                                                    int nnz = masks[
                                                            bm_i*64*576 +
                                                            mbrow_i*32*576 +
                                                            mbrow_i2*16*576 +
                                                            brow_i*4*576 +
                                                            mcol_i*16*mcol_kk +
                                                            mbrow_ii*2*576 +
                                                            mcol_ii*16 +
                                                            mbrow_iii*576 +
                                                            index];

                                                    if(nnz != 0){
                                                        indexes[
                                                            mbrow_iii*2 +
                                                            mcol_ii*2*2 +
                                                            pos] = n_i;

                                                        values[
                                                            mcol_ii*2*2 +
                                                            mbrow_iii*2 +
                                                            pos] =
                                                        sparse[
                                                            bm_i*64*576 +
                                                            mbrow_i*32*576 +
                                                            mbrow_i2*16*576 +
                                                            brow_i*4*576 +
                                                            mcol_i*16*mcol_kk +
                                                            mbrow_ii*2*576 +
                                                            mcol_ii*16 +
                                                            mbrow_iii*576 +
                                                            index];

                                                        pos+=1;
                                                    }
                                                } else {
                                                    if(n_i<2){
                                                        indexes[
                                                            mbrow_iii*2 +
                                                            mcol_ii*2*2 +
                                                            pos] = 0;

                                                        values[
                                                            mcol_ii*2*2 +
                                                            mbrow_iii*2 +
                                                            pos] = 0;

                                                        pos+=1;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    // write metadata
                                    unsigned int meta=0;
                                    for(int mbrow_iii=0; mbrow_iii<2; mbrow_iii++){
                                        for(int mcol_ii=0; mcol_ii<mcol_kk; mcol_ii++){
                                            for (int n_i=0; n_i<2; n_i++) {

                                                int idx = bm_i*64*80 +
                                                        mbrow_i*32*80+
                                                        mbrow_i2*16*80+
                                                        brow_i*4*16/2+
                                                        mcol_i*16*16/2 +
                                                        mbrow_ii*2*2 +
                                                        mcol_ii*2*4 +
                                                        mbrow_iii*2 +
                                                        n_i;

                                                max_idx = (idx>max_idx)?(idx):(max_idx);

                                                hA_values[
                                                        idx] =
                                                values[
                                                    mcol_ii*2*2 +
                                                    mbrow_iii*2 +
                                                    n_i];

                                                unsigned int tmp = indexes[
                                                            mbrow_iii*2 +
                                                            mcol_ii*2*2 +
                                                            n_i];
                                                meta |= (tmp << (mbrow_iii*(16/2)*2+mcol_ii*2*2+n_i*2));
                                            }
                                        }
                                    }
                                    hA_metadata[bm_i*mcol_k*64/2 +
                                                mbrow_i*mcol_k*32/2 +
                                                mbrow_i2*16/2 +
                                                brow_i*4/2  +
                                                mcol_i*32/2 +
                                                mbrow_ii] = meta;
                                }
                            }
                        }
                    }
                }
            }
            //cout << "max_idx: " << max_idx << endl;
        }
        