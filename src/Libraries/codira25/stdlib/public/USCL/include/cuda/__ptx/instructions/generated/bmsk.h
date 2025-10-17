/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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

#ifndef _CUDA_PTX_GENERATED_BMSK_H_
#define _CUDA_PTX_GENERATED_BMSK_H_

/*
// bmsk.clamp.b32 dest, a_reg, b_reg; // PTX ISA 76, SM_70
template <typename = void>
__device__ static inline uint32_t bmsk_clamp(
  uint32_t a_reg,
  uint32_t b_reg);
*/
#if __cccl_ptx_isa >= 760
extern "C" _CCCL_DEVICE void __cuda_ptx_bmsk_clamp_is_not_supported_before_SM_70__();
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint32_t bmsk_clamp(::cuda::std::uint32_t __a_reg, ::cuda::std::uint32_t __b_reg)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint32_t __dest;
  asm("bmsk.clamp.b32 %0, %1, %2;" : "=r"(__dest) : "r"(__a_reg), "r"(__b_reg) :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_bmsk_clamp_is_not_supported_before_SM_70__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 760

/*
// bmsk.wrap.b32 dest, a_reg, b_reg; // PTX ISA 76, SM_70
template <typename = void>
__device__ static inline uint32_t bmsk_wrap(
  uint32_t a_reg,
  uint32_t b_reg);
*/
#if __cccl_ptx_isa >= 760
extern "C" _CCCL_DEVICE void __cuda_ptx_bmsk_wrap_is_not_supported_before_SM_70__();
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint32_t bmsk_wrap(::cuda::std::uint32_t __a_reg, ::cuda::std::uint32_t __b_reg)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 700
  ::cuda::std::uint32_t __dest;
  asm("bmsk.wrap.b32 %0, %1, %2;" : "=r"(__dest) : "r"(__a_reg), "r"(__b_reg) :);
  return __dest;
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_bmsk_wrap_is_not_supported_before_SM_70__();
  return 0;
#  endif
}
#endif // __cccl_ptx_isa >= 760

#endif // _CUDA_PTX_GENERATED_BMSK_H_
