/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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
//
// included by:
//   ld to convert function variant name(s) to required flag bits
//   dyld to initialize actual flags bits
//
// FUNCTION_VARIANT_X86_64(_reqFlagBitNum, _name, _flagBitsInitialization)
//


//
//  x86_64 processor specific function variants
//
FUNCTION_VARIANT_X86_64( 0, "default",    true  )

// These were added in Spring 2025
FUNCTION_VARIANT_X86_64( 1, "sse41",      (*(uint64_t*)_COMM_PAGE_CPU_CAPABILITIES64) & kHasSSE4_1)
FUNCTION_VARIANT_X86_64( 2, "fma",        (*(uint64_t*)_COMM_PAGE_CPU_CAPABILITIES64) & kHasFMA)
FUNCTION_VARIANT_X86_64( 3, "avx",        (*(uint64_t*)_COMM_PAGE_CPU_CAPABILITIES64) & kHasAVX1_0)
FUNCTION_VARIANT_X86_64( 4, "avx2",       (*(uint64_t*)_COMM_PAGE_CPU_CAPABILITIES64) & kHasAVX2_0)
FUNCTION_VARIANT_X86_64( 5, "avx512f",    (*(uint64_t*)_COMM_PAGE_CPU_CAPABILITIES64) & kHasAVX512F)
FUNCTION_VARIANT_X86_64( 6, "avx512bw",   (*(uint64_t*)_COMM_PAGE_CPU_CAPABILITIES64) & kHasAVX512BW)
FUNCTION_VARIANT_X86_64( 7, "bmi1",       (*(uint64_t*)_COMM_PAGE_CPU_CAPABILITIES64) & kHasBMI1)
FUNCTION_VARIANT_X86_64( 8, "rosetta",    config.process.isTranslated)
FUNCTION_VARIANT_X86_64( 9, "haswell",    (_COMM_PAGE_CPUFAMILY == CPUFAMILY_INTEL_HASWELL))
FUNCTION_VARIANT_X86_64(10, "ivybridge",  true)
FUNCTION_VARIANT_X86_64(11, "nehalem",    true)

// These were added in Fall 2025



