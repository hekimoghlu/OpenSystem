/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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
#ifndef __CLANG_OPENMP_TIME_H__
#define __CLANG_OPENMP_TIME_H__

#ifndef _OPENMP
#error "This file is for OpenMP compilation only."
#endif

#if defined(__cplusplus)
#define __DEVICE__ static constexpr __attribute__((always_inline, nothrow))
#else
#define __DEVICE__ static __attribute__((always_inline, nothrow))
#endif

#include_next <time.h>

#pragma omp begin declare variant match(                                       \
    device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})

__DEVICE__ clock_t clock() { return __nvvm_read_ptx_sreg_clock(); }

#pragma omp end declare variant

#endif
