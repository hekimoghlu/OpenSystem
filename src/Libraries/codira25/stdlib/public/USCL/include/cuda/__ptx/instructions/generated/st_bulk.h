/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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

#ifndef _CUDA_PTX_GENERATED_ST_BULK_H_
#define _CUDA_PTX_GENERATED_ST_BULK_H_

/*
// st.bulk.weak.shared::cta [addr], size, initval; // PTX ISA 86, SM_100
template <int N32>
__device__ static inline void st_bulk(
  void* addr,
  uint64_t size,
  cuda::ptx::n32_t<N32> initval);
*/
#if __cccl_ptx_isa >= 860
extern "C" _CCCL_DEVICE void __cuda_ptx_st_bulk_is_not_supported_before_SM_100__();
template <int _N32>
_CCCL_DEVICE static inline void st_bulk(void* __addr, ::cuda::std::uint64_t __size, ::cuda::ptx::n32_t<_N32> __initval)
{
#  if _CCCL_CUDA_COMPILER(NVHPC) || __CUDA_ARCH__ >= 1000
  asm("st.bulk.weak.shared::cta [%0], %1, %2;"
      :
      : "r"(__as_ptr_smem(__addr)), "l"(__size), "n"(__initval.value)
      : "memory");
#  else
  // Unsupported architectures will have a linker error with a semi-decent error message
  __cuda_ptx_st_bulk_is_not_supported_before_SM_100__();
#  endif
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_ST_BULK_H_
