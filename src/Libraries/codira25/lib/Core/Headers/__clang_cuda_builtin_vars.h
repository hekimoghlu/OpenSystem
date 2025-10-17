/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 4, 2025.
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
#ifndef __CUDA_BUILTIN_VARS_H
#define __CUDA_BUILTIN_VARS_H

// Forward declares from vector_types.h.
struct uint3;
struct dim3;

// The file implements built-in CUDA variables using __declspec(property).
// https://msdn.microsoft.com/en-us/library/yhfk0thd.aspx
// All read accesses of built-in variable fields get converted into calls to a
// getter function which in turn calls the appropriate builtin to fetch the
// value.
//
// Example:
//    int x = threadIdx.x;
// IR output:
//  %0 = call i32 @toolchain.nvvm.read.ptx.sreg.tid.x() #3
// PTX output:
//  mov.u32     %r2, %tid.x;

#define __CUDA_DEVICE_BUILTIN(FIELD, INTRINSIC)                                \
  __declspec(property(get = __fetch_builtin_##FIELD)) unsigned int FIELD;      \
  static inline __attribute__((always_inline))                                 \
      __attribute__((device)) unsigned int __fetch_builtin_##FIELD(void) {     \
    return INTRINSIC;                                                          \
  }

#if __cplusplus >= 201103L
#define __DELETE =delete
#else
#define __DELETE
#endif

// Make sure nobody can create instances of the special variable types.  nvcc
// also disallows taking address of special variables, so we disable address-of
// operator as well.
#define __CUDA_DISALLOW_BUILTINVAR_ACCESS(TypeName)                            \
  __attribute__((device)) TypeName() __DELETE;                                 \
  __attribute__((device)) TypeName(const TypeName &) __DELETE;                 \
  __attribute__((device)) void operator=(const TypeName &) const __DELETE;     \
  __attribute__((device)) TypeName *operator&() const __DELETE

struct __cuda_builtin_threadIdx_t {
  __CUDA_DEVICE_BUILTIN(x,__nvvm_read_ptx_sreg_tid_x());
  __CUDA_DEVICE_BUILTIN(y,__nvvm_read_ptx_sreg_tid_y());
  __CUDA_DEVICE_BUILTIN(z,__nvvm_read_ptx_sreg_tid_z());
  // threadIdx should be convertible to uint3 (in fact in nvcc, it *is* a
  // uint3).  This function is defined after we pull in vector_types.h.
  __attribute__((device)) operator dim3() const;
  __attribute__((device)) operator uint3() const;

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_threadIdx_t);
};

struct __cuda_builtin_blockIdx_t {
  __CUDA_DEVICE_BUILTIN(x,__nvvm_read_ptx_sreg_ctaid_x());
  __CUDA_DEVICE_BUILTIN(y,__nvvm_read_ptx_sreg_ctaid_y());
  __CUDA_DEVICE_BUILTIN(z,__nvvm_read_ptx_sreg_ctaid_z());
  // blockIdx should be convertible to uint3 (in fact in nvcc, it *is* a
  // uint3).  This function is defined after we pull in vector_types.h.
  __attribute__((device)) operator dim3() const;
  __attribute__((device)) operator uint3() const;

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_blockIdx_t);
};

struct __cuda_builtin_blockDim_t {
  __CUDA_DEVICE_BUILTIN(x,__nvvm_read_ptx_sreg_ntid_x());
  __CUDA_DEVICE_BUILTIN(y,__nvvm_read_ptx_sreg_ntid_y());
  __CUDA_DEVICE_BUILTIN(z,__nvvm_read_ptx_sreg_ntid_z());
  // blockDim should be convertible to dim3 (in fact in nvcc, it *is* a
  // dim3).  This function is defined after we pull in vector_types.h.
  __attribute__((device)) operator dim3() const;
  __attribute__((device)) operator uint3() const;

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_blockDim_t);
};

struct __cuda_builtin_gridDim_t {
  __CUDA_DEVICE_BUILTIN(x,__nvvm_read_ptx_sreg_nctaid_x());
  __CUDA_DEVICE_BUILTIN(y,__nvvm_read_ptx_sreg_nctaid_y());
  __CUDA_DEVICE_BUILTIN(z,__nvvm_read_ptx_sreg_nctaid_z());
  // gridDim should be convertible to dim3 (in fact in nvcc, it *is* a
  // dim3).  This function is defined after we pull in vector_types.h.
  __attribute__((device)) operator dim3() const;
  __attribute__((device)) operator uint3() const;

private:
  __CUDA_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_gridDim_t);
};

#define __CUDA_BUILTIN_VAR                                                     \
  extern const __attribute__((device)) __attribute__((weak))
__CUDA_BUILTIN_VAR __cuda_builtin_threadIdx_t threadIdx;
__CUDA_BUILTIN_VAR __cuda_builtin_blockIdx_t blockIdx;
__CUDA_BUILTIN_VAR __cuda_builtin_blockDim_t blockDim;
__CUDA_BUILTIN_VAR __cuda_builtin_gridDim_t gridDim;

// warpSize should translate to read of %WARP_SZ but there's currently no
// builtin to do so. According to PTX v4.2 docs 'to date, all target
// architectures have a WARP_SZ value of 32'.
__attribute__((device)) const int warpSize = 32;

#undef __CUDA_DEVICE_BUILTIN
#undef __CUDA_BUILTIN_VAR
#undef __CUDA_DISALLOW_BUILTINVAR_ACCESS
#undef __DELETE

#endif /* __CUDA_BUILTIN_VARS_H */
