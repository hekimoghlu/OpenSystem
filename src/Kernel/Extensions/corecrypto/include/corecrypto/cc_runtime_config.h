/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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
#ifndef CORECRYPTO_CC_RUNTIME_CONFIG_H_
#define CORECRYPTO_CC_RUNTIME_CONFIG_H_

#include <corecrypto/cc_config.h>

/* Only intel systems have these runtime switches today. */

#if defined(__x86_64__) || defined(__i386__)

#if CC_KERNEL
    #include <i386/cpuid.h>
    #define CC_HAS_RDRAND() ((cpuid_features() & CPUID_FEATURE_RDRAND) != 0)
#elif CC_XNU_KERNEL_AVAILABLE
    #include <System/i386/cpu_capabilities.h>

    extern int _cpu_capabilities;
    #define CC_HAS_RDRAND() (_cpu_capabilities & kHasRDRAND)
#else
    #define CC_HAS_RDRAND() 0
#endif

#if (CCSHA1_VNG_INTEL || CCSHA2_VNG_INTEL || CCAES_INTEL_ASM)

#if CC_KERNEL
    #include <i386/cpuid.h>
    #define CC_HAS_AESNI() ((cpuid_features() & CPUID_FEATURE_AES) != 0)
    #define CC_HAS_SupplementalSSE3() ((cpuid_features() & CPUID_FEATURE_SSSE3) != 0)
    #define CC_HAS_AVX1() ((cpuid_features() & CPUID_FEATURE_AVX1_0) != 0)
    #define CC_HAS_AVX2() ((cpuid_info()->cpuid_leaf7_features & CPUID_LEAF7_FEATURE_AVX2) != 0)
    #define CC_HAS_AVX512_AND_IN_KERNEL()    ((cpuid_info()->cpuid_leaf7_features & CPUID_LEAF7_FEATURE_AVX512F) !=0)

#elif CC_XNU_KERNEL_AVAILABLE
    #include <System/i386/cpu_capabilities.h>

    extern int _cpu_capabilities;
    #define CC_HAS_AESNI() (_cpu_capabilities & kHasAES)
    #define CC_HAS_SupplementalSSE3() (_cpu_capabilities & kHasSupplementalSSE3)
    #define CC_HAS_AVX1() (_cpu_capabilities & kHasAVX1_0)
    #define CC_HAS_AVX2() (_cpu_capabilities & kHasAVX2_0)
    #define CC_HAS_AVX512_AND_IN_KERNEL() 0
#else
    #define CC_HAS_AESNI() 0
    #define CC_HAS_SupplementalSSE3() 0
    #define CC_HAS_AVX1() 0
    #define CC_HAS_AVX2() 0
    #define CC_HAS_AVX512_AND_IN_KERNEL()  0
#endif

#endif  // (CCSHA1_VNG_INTEL || CCSHA2_VNG_INTEL || CCAES_INTEL_ASM)

#endif  // defined(__x86_64__) || defined(__i386__)

#endif /* CORECRYPTO_CC_RUNTIME_CONFIG_H_ */
