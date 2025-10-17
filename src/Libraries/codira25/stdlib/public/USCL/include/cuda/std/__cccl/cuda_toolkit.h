/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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
#ifndef __CCCL_CUDA_TOOLKIT_H
#define __CCCL_CUDA_TOOLKIT_H

#include <uscl/std/__cccl/compiler.h>
#include <uscl/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION() || _CCCL_HAS_INCLUDE(<cuda_runtime_api.h>)
#  define _CCCL_HAS_CTK() 1
#else // ^^^ has cuda toolkit ^^^ / vvv no cuda toolkit vvv
#  define _CCCL_HAS_CTK() 0
#endif // ^^^ no cuda toolkit ^^^

// CUDA compilers preinclude cuda_runtime.h, so we need to include it here to get the CUDART_VERSION macro
#if _CCCL_HAS_CTK() && !_CCCL_CUDA_COMPILATION()
#  include <cuda_runtime_api.h>
#endif // _CCCL_HAS_CTK() && !_CCCL_CUDA_COMPILATION()

// Check compatibility of the CUDA compiler and CUDA toolkit headers
#if _CCCL_CUDA_COMPILATION()
#  if !_CCCL_CUDACC_EQUAL((CUDART_VERSION / 1000), (CUDART_VERSION % 1000) / 10)
#    error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
#  endif // !_CCCL_CUDACC_EQUAL((CUDART_VERSION / 1000), (CUDART_VERSION % 1000) / 10)
#endif // _CCCL_CUDA_COMPILATION()

#if _CCCL_HAS_CTK()
#  define _CCCL_CTK() (CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10)
#else // ^^^ has cuda toolkit ^^^ / vvv no cuda toolkit vvv
#  define _CCCL_CTK() _CCCL_VERSION_INVALID()
#endif // ^^^ no cuda toolkit ^^^

#define _CCCL_CTK_MAKE_VERSION(_MAJOR, _MINOR) ((_MAJOR) * 1000 + (_MINOR) * 10)
#define _CCCL_CTK_BELOW(...)                   _CCCL_VERSION_COMPARE(_CCCL_CTK_, _CCCL_CTK, <, __VA_ARGS__)
#define _CCCL_CTK_AT_LEAST(...)                _CCCL_VERSION_COMPARE(_CCCL_CTK_, _CCCL_CTK, >=, __VA_ARGS__)

#endif // __CCCL_CUDA_TOOLKIT_H
