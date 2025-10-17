/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 20, 2023.
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

// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_EXIT_H_
#define _CUDA_PTX_GENERATED_EXIT_H_

/*
// exit; // PTX ISA 10, SM_50
template <typename = void>
__device__ static inline void exit();
*/
#if __cccl_ptx_isa >= 100
extern "C" _CCCL_DEVICE void __cuda_ptx_exit_is_not_supported_before_SM_50__();
template <typename = void>
_CCCL_DEVICE static inline void exit()
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 500
  asm volatile("exit;" : : :);
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_exit_is_not_supported_before_SM_50__();
#  endif
}
#endif // __cccl_ptx_isa >= 100

#endif // _CUDA_PTX_GENERATED_EXIT_H_
