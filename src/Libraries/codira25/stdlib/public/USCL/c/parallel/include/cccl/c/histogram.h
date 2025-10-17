/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef CCCL_C_EXPERIMENTAL
#  error "C exposure is experimental and subject to change. Define CCCL_C_EXPERIMENTAL to acknowledge this notice."
#endif // !CCCL_C_EXPERIMENTAL

#include <cuda.h>
#include <stdbool.h>
#include <stdint.h>

#include <cccl/c/extern_c.h>
#include <cccl/c/types.h>

CCCL_C_EXTERN_C_BEGIN

typedef struct cccl_device_histogram_build_result_t
{
  int cc;
  void* cubin;
  size_t cubin_size;
  CUlibrary library;
  cccl_type_info counter_type;
  int num_active_channels;
  CUkernel init_kernel;
  CUkernel sweep_kernel;
} cccl_device_histogram_build_result_t;

CCCL_C_API CUresult cccl_device_histogram_build(
  cccl_device_histogram_build_result_t* build,
  int num_channels,
  int num_active_channels,
  cccl_iterator_t d_samples,
  int num_output_levels_val,
  cccl_iterator_t d_output_histograms,
  cccl_value_t lower_level,
  int64_t num_rows,
  int64_t row_stride_samples,
  bool is_evenly_segmented,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path);

// Extended version with build configuration
CCCL_C_API CUresult cccl_device_histogram_build_ex(
  cccl_device_histogram_build_result_t* build,
  int num_channels,
  int num_active_channels,
  cccl_iterator_t d_samples,
  int num_output_levels_val,
  cccl_iterator_t d_output_histograms,
  cccl_value_t lower_level,
  int64_t num_rows,
  int64_t row_stride_samples,
  bool is_evenly_segmented,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config);

CCCL_C_API CUresult cccl_device_histogram_even(
  cccl_device_histogram_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_samples,
  cccl_iterator_t d_output_histograms,
  cccl_value_t num_output_levels,
  cccl_value_t lower_level,
  cccl_value_t upper_level,
  int64_t num_row_pixels,
  int64_t num_rows,
  int64_t row_stride_samples,
  CUstream stream);

CCCL_C_API CUresult cccl_device_histogram_cleanup(cccl_device_histogram_build_result_t* bld_ptr);

CCCL_C_EXTERN_C_END
