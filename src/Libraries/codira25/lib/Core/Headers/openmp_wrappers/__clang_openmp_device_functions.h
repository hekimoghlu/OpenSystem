/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 28, 2021.
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
#ifndef __CLANG_OPENMP_DEVICE_FUNCTIONS_H__
#define __CLANG_OPENMP_DEVICE_FUNCTIONS_H__

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __NVPTX__
#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

#pragma push_macro("__CUDA__")
#define __CUDA__
#define __OPENMP_NVPTX__

/// Include declarations for libdevice functions.
#include <__clang_cuda_libdevice_declares.h>

/// Provide definitions for these functions.
#include <__clang_cuda_device_functions.h>

#undef __OPENMP_NVPTX__
#pragma pop_macro("__CUDA__")

#pragma omp end declare variant
#endif

#ifdef __AMDGCN__
#pragma omp begin declare variant match(device = {arch(amdgcn)})

// Import types which will be used by __clang_hip_libdevice_declares.h
#ifndef __cplusplus
#include <stdint.h>
#endif

#define __OPENMP_AMDGCN__
#pragma push_macro("__device__")
#define __device__

/// Include declarations for libdevice functions.
#include <__clang_hip_libdevice_declares.h>

#pragma pop_macro("__device__")
#undef __OPENMP_AMDGCN__

#pragma omp end declare variant
#endif

#ifdef __cplusplus
} // extern "C"
#endif

// Ensure we make `_ZdlPv`, aka. `operator delete(void*)` available without the
// need to `include <new>` in C++ mode.
#ifdef __cplusplus

// We require malloc/free.
#include <cstdlib>

#pragma push_macro("OPENMP_NOEXCEPT")
#if __cplusplus >= 201103L
#define OPENMP_NOEXCEPT noexcept
#else
#define OPENMP_NOEXCEPT
#endif

// Device overrides for non-placement new and delete.
inline void *operator new(__SIZE_TYPE__ size) {
  if (size == 0)
    size = 1;
  return ::malloc(size);
}

inline void *operator new[](__SIZE_TYPE__ size) { return ::operator new(size); }

inline void operator delete(void *ptr)OPENMP_NOEXCEPT { ::free(ptr); }

inline void operator delete[](void *ptr) OPENMP_NOEXCEPT {
  ::operator delete(ptr);
}

// Sized delete, C++14 only.
#if __cplusplus >= 201402L
inline void operator delete(void *ptr, __SIZE_TYPE__ size)OPENMP_NOEXCEPT {
  ::operator delete(ptr);
}
inline void operator delete[](void *ptr, __SIZE_TYPE__ size) OPENMP_NOEXCEPT {
  ::operator delete(ptr);
}
#endif

#pragma pop_macro("OPENMP_NOEXCEPT")
#endif

#endif
