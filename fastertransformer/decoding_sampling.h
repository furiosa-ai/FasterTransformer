/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * Decoder transformer
 **/

#pragma once

#include "fastertransformer/utils/common.h"
#include "fastertransformer/utils/functions.h"
#include "fastertransformer/utils/allocator.h"
#include "fastertransformer/open_decoder.h"
#include "fastertransformer/cuda/cuda_kernels.h"
#include "fastertransformer/utils/arguments.h"

#include "fastertransformer/utils/nvtx_utils.h"

#include <cuda_runtime.h>

namespace fastertransformer
{

template <OperationType OpType_>
class DecodingSampling
{
private:
  typedef DecoderTransformerTraits<OpType_> Traits_;
  typedef typename Traits_::DataType DataType_;
  const IAllocator &allocator_;
  struct DecodingSamplingArguments args_;

  const cudaDataType_t computeType_ = Traits_::computeType;
  const cudaDataType_t AType_ = Traits_::AType;
  const cudaDataType_t BType_ = Traits_::BType;
  const cudaDataType_t CType_ = Traits_::CType;
  std::map<std::string, cublasLtMatmulAlgo_info> cublasAlgoMap_;

  OpenDecoder<OpType_> *decoder_;
  DataType_ **K_cache_;
  DataType_ **V_cache_;
  DataType_ **K_mem_cache_;
  DataType_ **V_mem_cache_;
  DataType_ *from_tensor_[2];
  DataType_ *decoder_buf_;
  DataType_ *decoder_normed_result_buf_;
  DataType_ *logits_buf_;
  int *word_ids_buf_;
  bool *finished_buf_;

  void *buf_;
  int *finished_count_buf_;
  bool *h_finished_buf_;

  void *topk_workspace_ = nullptr;
  size_t topk_workspace_size_ = 0;
  void *topp_workspace_ = nullptr;
  size_t topp_workspace_size_ = 0;
  void *cublas_workspace_ = nullptr;
  curandState_t *curandstate_buf_; 
  int *topp_id_vals_buf_;
  int *topp_offset_buf_;
  int *begin_topp_offset_buf_;

  DataType_ *padded_embedding_kernel;
  DataType_ *padded_embedding_bias;

public:
  DecodingSampling(const IAllocator &allocator, const int batch_size,
                   const int seq_len,
                   const int head_num, const int size_per_head,
                   const int vocab_size, const int decoder_layers,
                   const int memory_hidden_units, const int memory_max_seq_len,
                   const int start_id, const int end_id,
                   const int candidate_num = 0,
                   const float probability_threshold = 0.0,
                   const int is_fuse_qkv = false) : allocator_(allocator)
  {
    args_.batch_size_ = batch_size;
    args_.seq_len_ = seq_len;
    args_.head_num_ = head_num;
    args_.size_per_head_ = size_per_head;
    args_.hidden_units_ = head_num * size_per_head;
    args_.decoder_layers_ = decoder_layers;
    args_.vocab_size_ = vocab_size;
    if(std::is_same<DataType_, float>::value)
      args_.vocab_size_padded_ = vocab_size;
    else if(std::is_same<DataType_, half>::value)
      args_.vocab_size_padded_ = (int)(ceil(vocab_size / 8.)) * 8;

    args_.candidate_num_ = candidate_num;
    args_.probability_threshold_ = probability_threshold;
    args_.start_id_ = start_id;
    args_.end_id_ = end_id;

    if (args_.candidate_num_ == 0 && args_.probability_threshold_ == 0.0)
    {
      printf("[ERROR] Candidate_num for topk is 0 and probability threshold for top p is 0.0 \n");
      exit(-1);
    }
    else if (args_.candidate_num_ != 0 && args_.probability_threshold_ != 0.0)
    {
      printf("[ERROR] Candidate_num for topk is not 0 and probability threshold for top p is not 0.0 \n");
      exit(-1);
    }
#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    K_cache_ = new DataType_ *[1];
    V_cache_ = new DataType_ *[1];

    K_mem_cache_ = new DataType_ *[args_.decoder_layers_];
    V_mem_cache_ = new DataType_ *[args_.decoder_layers_];

    decoder_ = new OpenDecoder<OpType_>(head_num, size_per_head, memory_hidden_units, is_fuse_qkv);
    decoder_->set_max_batch_size(batch_size);

    size_t from_tensor_size = args_.batch_size_ * args_.hidden_units_;                    // type T
    size_t decoder_workspace_size = decoder_->getWorkspaceSize();                         // type T
    size_t decoder_normed_result_buffer_size = args_.batch_size_ * args_.hidden_units_;   // type T
    size_t cache_size = args_.batch_size_ * args_.seq_len_ * args_.hidden_units_;         // type T
    size_t mem_cache_size = args_.batch_size_ * memory_max_seq_len * args_.hidden_units_; // type T
    size_t logits_buf_size = args_.batch_size_ * args_.vocab_size_padded_; // type T

    size_t word_ids_buf_size = args_.batch_size_;                   //type int
    size_t finished_buf_size = args_.batch_size_;                   //type bool
    size_t finished_count_size = (size_t)(ceil(1 / 32.)) * 32;         // type int

    size_t topp_id_vals_buf_size = args_.batch_size_ * args_.vocab_size_padded_; // type int
    size_t topp_offset_buf_size = args_.batch_size_ + 1; // type int
    size_t begin_topp_offset_buf_size = topp_offset_buf_size;
    size_t curandState_size = args_.batch_size_;
    size_t padded_embedding_kernel_size = args_.hidden_units_ * args_.vocab_size_padded_;
    size_t padded_embedding_bias_size = args_.vocab_size_padded_;
    if(std::is_same<DataType_, float>::value || (std::is_same<DataType_, half>::value && args_.vocab_size_ == args_.vocab_size_padded_))
    {
      padded_embedding_kernel_size = 0;
      padded_embedding_bias_size = 0;
    }

    // prevent memory misalinged address
    logits_buf_size = (size_t)(ceil(logits_buf_size / 4.)) * 4;
    word_ids_buf_size = (size_t)(ceil(word_ids_buf_size / 4.)) * 4;
    finished_buf_size = (size_t)(ceil(finished_buf_size / 32.)) * 32;

    topp_id_vals_buf_size = (size_t)(ceil(topp_id_vals_buf_size / 4.)) * 4;
    topp_offset_buf_size = (size_t)(ceil(topp_offset_buf_size / 4.)) * 4;
    begin_topp_offset_buf_size = topp_offset_buf_size;
    
    topP_sampling_kernel_kernelLauncher_v2(topp_workspace_,
                                        topp_workspace_size_,
                                        logits_buf_,
                                        topp_id_vals_buf_,
                                        topp_offset_buf_,
                                        begin_topp_offset_buf_,
                                        finished_buf_,
                                        curandstate_buf_,
                                        args_,
                                        nullptr, 
                                        nullptr, 
                                        args_.vocab_size_padded_,
                                        0,
                                        args_.batch_size_);

    topK_sampling_kernel_kernelLauncher_v2(topk_workspace_,
                                           topk_workspace_size_,
                                           logits_buf_,
                                           nullptr,
                                           nullptr,
                                           finished_buf_,
                                           curandstate_buf_,
                                           args_,
                                           0,
                                           args_.batch_size_);

    size_t datatype_buf_size = from_tensor_size * 2 + decoder_workspace_size +
                            (cache_size * 4 + mem_cache_size * 2) * args_.decoder_layers_ + decoder_normed_result_buffer_size;

    buf_ = reinterpret_cast<void *>(allocator_.malloc(
        ( (sizeof(DataType_) == sizeof(half)) ? CUBLAS_WORKSPACE_SIZE : 0 ) + 
        sizeof(DataType_) * (datatype_buf_size + logits_buf_size) +
        sizeof(DataType_) * (padded_embedding_kernel_size + padded_embedding_bias_size) +
        sizeof(int) * word_ids_buf_size +
        sizeof(bool) * finished_buf_size +
        sizeof(int) * finished_count_size +
        sizeof(int) * (topp_id_vals_buf_size + 2 * topp_offset_buf_size) +
        topp_workspace_size_ + topk_workspace_size_  + curandState_size * sizeof(curandState_t)));
    
    if (sizeof(DataType_) == sizeof(half))
    {
      cublas_workspace_ = buf_;
      from_tensor_[0] = (DataType_ *)((char*)cublas_workspace_ + CUBLAS_WORKSPACE_SIZE);
    }
    else
    {
      cublas_workspace_ = nullptr;
      from_tensor_[0] = (DataType_ *)buf_;
    }
    from_tensor_[1] = (DataType_ *)(from_tensor_[0] + from_tensor_size);

    for (int i = 0; i < args_.decoder_layers_; ++i)
    {
      K_mem_cache_[i] = from_tensor_[1] + from_tensor_size + i * mem_cache_size * 2;
      V_mem_cache_[i] = from_tensor_[1] + from_tensor_size + i * mem_cache_size * 2 + mem_cache_size;
    }

    /* We use two-way buffer since we have to update KV buf at the end of each step. */
    K_cache_[0] = V_mem_cache_[args_.decoder_layers_ - 1] + mem_cache_size + 0 * cache_size * args_.decoder_layers_;
    V_cache_[0] = V_mem_cache_[args_.decoder_layers_ - 1] + mem_cache_size + 1 * cache_size * args_.decoder_layers_;

    decoder_buf_ = V_cache_[0] + cache_size * args_.decoder_layers_;
    decoder_normed_result_buf_ = (decoder_buf_ + decoder_workspace_size);
    logits_buf_ = decoder_normed_result_buf_ + decoder_normed_result_buffer_size;
    word_ids_buf_ = (int *)(logits_buf_ + logits_buf_size);
    finished_buf_ = (bool *)(word_ids_buf_ + word_ids_buf_size);
    finished_count_buf_ = (int *)(finished_buf_ + finished_buf_size);
    topp_id_vals_buf_ = (int *)(finished_count_buf_ + finished_count_size);
    begin_topp_offset_buf_ = (int *)(topp_id_vals_buf_ + topp_id_vals_buf_size);
    topp_offset_buf_ = (int *)(begin_topp_offset_buf_ + begin_topp_offset_buf_size);
    topp_workspace_ = (void*)(topp_offset_buf_ + topp_offset_buf_size);
    topk_workspace_ = (void*)((char*)topp_workspace_ + topp_workspace_size_);
    padded_embedding_kernel = (DataType_*)((char*)topk_workspace_ + topk_workspace_size_);
    padded_embedding_bias = (DataType_*)(padded_embedding_kernel + padded_embedding_kernel_size);
    curandstate_buf_ = (curandState_t*)(padded_embedding_bias + padded_embedding_bias_size);

    h_finished_buf_ = new bool[finished_buf_size];

    int isConfigExist = access("decoding_gemm_config.in", 0);
    if (isConfigExist == -1)
    {
      printf("[WARNING] decoding_gemm_config.in is not found\n");
    }
    else
    {
      readAlgoFromConfig(cublasAlgoMap_, 1);
      // check that the gemm_config setting is runnable
      for (auto iter = cublasAlgoMap_.begin() ; iter != cublasAlgoMap_.end() ; iter++)
      {
        int algoId = iter->second.algoId;
        int stages = iter->second.stages;
        //only check for cublas
        if (stages != -1)
          continue;
        if (Traits_::OpType == OperationType::FP32)
        {
          if (algoId > CUBLAS_GEMM_ALGO23 || algoId < CUBLAS_GEMM_DEFAULT)
          {
            // the algorithm is not for FP32
            printf("[ERROR] cuBLAS Algorithm %d is not used in FP32. \n", algoId);
            exit(-1);
          }
        }
        else
        {
          if (algoId > CUBLAS_GEMM_ALGO15_TENSOR_OP || algoId < CUBLAS_GEMM_DEFAULT_TENSOR_OP)
          {
            // the algorithm is not for FP16
            printf("[ERROR] cuBLAS Algorithm %d is not used in FP16. \n", algoId);
            exit(-1);
          }
        }
      }
    }
  }

  void forward(const DecoderInitParam<DataType_> *param,
               DecodingInitParam<DataType_> decoding_params)
  {

    PUSH_RANGE("Entire Forward")    //mgwg

#ifndef NDEBUG
    PRINT_FUNC_NAME_();
#endif
    const int m = args_.batch_size_;
    const int k = args_.hidden_units_;
    const int n = args_.vocab_size_padded_;
    const DataType_* embedding_kernel_ptr = nullptr;
    const DataType_* embedding_bias_ptr = nullptr;

    /*
      sequence_length initialize to 0
      finished: false
      word_ids: start_id_
    */

    if (args_.candidate_num_ != 0)
    {
      sampling_init_kernelLauncher(finished_buf_, decoding_params.sequence_length, word_ids_buf_, 
                                   args_.start_id_, args_.batch_size_, decoding_params.stream);
    }
    else if (args_.probability_threshold_ != 0.0)
    {
      topp_initialization_kernelLauncher_v2(finished_buf_,
                                            decoding_params.sequence_length,
                                            word_ids_buf_,
                                            topp_id_vals_buf_,
                                            topp_offset_buf_,
                                            begin_topp_offset_buf_,
                                            args_.vocab_size_padded_,
                                            args_,
                                            decoding_params.stream);
    }
    ker_curand_setupLauncher(curandstate_buf_, 
                             args_,
                             decoding_params.stream);

#ifndef NDEBUG
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
#endif

    if(std::is_same<DataType_, float>::value || (std::is_same<DataType_, half>::value && args_.vocab_size_ == args_.vocab_size_padded_))
    {
      embedding_kernel_ptr = (const DataType_ *)decoding_params.embedding_kernel;
      embedding_bias_ptr = (const DataType_ *)decoding_params.embedding_bias;  
    }
    else if(std::is_same<DataType_, half>::value)
    {
      kernel_padding_kernelLauncher(padded_embedding_kernel, decoding_params.embedding_kernel, args_.hidden_units_,
                                    args_.vocab_size_, args_.vocab_size_padded_, decoding_params.stream);

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      bias_padding_kernelLauncher(padded_embedding_bias, decoding_params.embedding_bias, 
                                  args_.vocab_size_, args_.vocab_size_padded_, decoding_params.stream);

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif
      embedding_kernel_ptr = padded_embedding_kernel;
      embedding_bias_ptr = padded_embedding_bias;
    }

    int cache_size = args_.batch_size_ * args_.seq_len_ * args_.hidden_units_; // type T

    for (uint step = 1; step <= args_.seq_len_; ++step)
    {

      PUSH_RANGE("one step")    //mgwg

      PUSH_RANGE("input embedding")    //mgwg

      embedding_lookup_sine_position_encoding_kernel_launcher(from_tensor_[0],
                                                              decoding_params.embedding_table,
                                                              decoding_params.position_encoding_table + (step - 1) * args_.hidden_units_,
                                                              word_ids_buf_,
                                                              args_.batch_size_,
                                                              args_.hidden_units_,
                                                              decoding_params.stream);

      POP_RANGE // "input embedding"   //mgwg

      int from_id, out_id;
      for (int layer = 0; layer < args_.decoder_layers_; ++layer)
      {
        /*
          For the first layer (layer-0), from_id is 0. We also stored the embedding lookup 
          result in from_tensor_[0]
        */
        from_id = layer & 0x1;
        out_id = 1 - from_id;

        /*
          We use one decoder_ object to process multiple decoder layers. 
        
          At the beginning of each decoder layer, we initialize the decoder object 
          with corresponding weights and decoder_buf_.

          The decoder_buf_ is reused.
        */
        decoder_->initialize(param[layer], decoder_buf_, cublas_workspace_);

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif

        PUSH_RANGE("one decoder layer")    //mgwg

        decoder_->forward(from_tensor_[from_id], decoding_params.memory_tensor,
                          K_cache_[0] + layer * cache_size,
                          V_cache_[0] + layer * cache_size,
                          K_mem_cache_[layer], V_mem_cache_[layer],
                          decoding_params.memory_sequence_length, from_tensor_[out_id], step, args_.seq_len_,
                          true, finished_buf_);

        POP_RANGE // "one decoder layer"   //mgwg

#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError());
#endif
      }

      PUSH_RANGE("layer_norm")    //mgwg

      layer_norm(from_tensor_[out_id], decoding_params.layernorm.gamma,
                 decoding_params.layernorm.beta, decoder_normed_result_buf_, m, k, decoding_params.stream);

      POP_RANGE // "layer_norm"   //mgwg
      
      DataType_ alpha = (DataType_)1.0f;
      DataType_ beta = (DataType_)0.0f;

      PUSH_RANGE("classifier")    //mgwg

      cublasMM_cublasLtMM_wrapper_decoder(decoding_params.cublaslt_handle, 
                                          decoding_params.cublas_handle, 
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          n, m, k,
                                          &alpha,
                                          embedding_kernel_ptr, AType_, n,
                                          decoder_normed_result_buf_, BType_, k,
                                          &beta,
                                          logits_buf_, CType_, n,
                                          decoding_params.stream, cublasAlgoMap_,
                                          cublas_workspace_);

      POP_RANGE // "classifier"   //mgwg

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      if (args_.candidate_num_ != 0)
      {

        PUSH_RANGE("update_logits_without_softmax")    //mgwg

        // top k sampling
        update_logits_without_softmax(logits_buf_,
                                      embedding_bias_ptr,
                                      args_.end_id_,
                                      finished_buf_,
                                      m, n, decoding_params.stream);

        POP_RANGE // "update_logits_without_softmax"   //mgwg

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

        PUSH_RANGE("topK_sampling")    //mgwg

        topK_sampling_kernel_kernelLauncher_v2(topk_workspace_,
                                               topk_workspace_size_,
                                               logits_buf_,
                                               decoding_params.output_ids + (step - 1) * args_.batch_size_,
                                               decoding_params.sequence_length,
                                               finished_buf_,
                                               curandstate_buf_, // used as random number
                                               args_,
                                               decoding_params.stream,
                                               args_.batch_size_);

        POP_RANGE // "topK_sampling"   //mgwg

      }
      else if (args_.probability_threshold_ != 0.0)
      {

        PUSH_RANGE("softmax")    //mgwg

        // top p sampling
        softmax_kernelLauncher(logits_buf_,
                               embedding_bias_ptr,
                               args_.end_id_,
                               finished_buf_,
                               m, n, n, decoding_params.stream);

        POP_RANGE // "softmax"   //mgwg

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

        PUSH_RANGE("topP_sampling")    //mgwg

        topP_sampling_kernel_kernelLauncher_v2(topp_workspace_,
                                               topp_workspace_size_,
                                               logits_buf_,
                                               topp_id_vals_buf_,
                                               topp_offset_buf_,
                                               begin_topp_offset_buf_,
                                               finished_buf_,
                                               curandstate_buf_,
                                               args_,
                                               decoding_params.output_ids + (step - 1) * args_.batch_size_,
                                               decoding_params.sequence_length,
                                               n,
                                               decoding_params.stream,
                                               args_.batch_size_);

        POP_RANGE // "topP_sampling"   //mgwg

      }

      word_ids_buf_ = decoding_params.output_ids + (step - 1) * args_.batch_size_;

#ifndef NDEBUG
      cudaDeviceSynchronize();
      check_cuda_error(cudaGetLastError());
#endif

      PUSH_RANGE("is_finished")    //mgwg

      // TODO Find a better method to check the is_finished
      cudaMemcpy(h_finished_buf_, finished_buf_, sizeof(bool) * args_.batch_size_, cudaMemcpyDeviceToHost);
      uint sum = 0;
      for (uint i = 0; i < args_.batch_size_; i++)
      {
        sum += (int)h_finished_buf_[i];
      }

      POP_RANGE // "is_finished"   //mgwg
      POP_RANGE // "one step"   //mgwg

      if (sum == args_.batch_size_)
        break;
    }
    POP_RANGE // "Entire Forward"   //mgwg
  }

  virtual ~DecodingSampling()
  {
    delete[] K_cache_;
    delete[] V_cache_;
    delete[] K_mem_cache_;
    delete[] V_mem_cache_;
    delete[] h_finished_buf_;
    delete decoder_;
    allocator_.free(buf_);
  }
};

} //namespace fastertransformer
