/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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

// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <uscl/__cccl_config>

#ifndef TEST_HALF_T
#  if _CCCL_HAS_NVFP16()
#    define TEST_HALF_T() 1
#  else
#    define TEST_HALF_T() 0
#  endif
#endif // TEST_HALF_T

#ifndef TEST_BF_T
#  if _CCCL_HAS_NVBF16()
#    define TEST_BF_T() 1
#  else
#    define TEST_BF_T() 0
#  endif
#endif // TEST_BF_T

#ifndef TEST_INT128
#  if _CCCL_HAS_INT128() && !_CCCL_CUDA_COMPILER(CLANG) // clang-cuda crashes with int128 in generator.cu
#    define TEST_INT128() 1
#  else
#    define TEST_INT128() 0
#  endif
#endif // TEST_INT128

#if TEST_HALF_T()
#  include <cuda_fp16.h>

#  include <c2h/half.cuh>
#endif // TEST_HALF_T()

#if TEST_BF_T()
#  include <cuda_bf16.h>

#  include <c2h/bfloat16.cuh>
#endif // TEST_BF_T()
