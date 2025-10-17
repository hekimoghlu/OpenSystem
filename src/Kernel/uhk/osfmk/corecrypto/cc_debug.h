/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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
//debug configuration header file
#ifndef _CORECRYPTO_CCN_DEBUG_H_
#define _CORECRYPTO_CCN_DEBUG_H_

#include <corecrypto/cc_config.h>

// DO NOT INCLUDE this HEADER file in CoreCrypto files added for XNU project or headers
// included by external clients.

// ========================
// Printf for corecrypto
// ========================
#if CC_KERNEL
    #include <pexpert/pexpert.h>
    #define cc_printf(x...) kprintf(x)
    #if !CONFIG_EMBEDDED
extern int printf(const char *format, ...) __cc_printflike(1, 2);
    #endif
#elif CC_IBOOT || CC_RTKIT || CC_RTKITROM
    #include <stdio.h>
    #define cc_printf(x...) printf(x)
#elif CC_SGX || CC_EFI
    #define cc_printf(x...)
#elif CC_TXM
    #define cc_printf(x...)
#elif CC_SPTM
    #define cc_printf(x...)
#else
    #include <stdio.h>
    #define cc_printf(x...) fprintf(stderr, x)
#endif

// ========================
// Integer types
// ========================

#if CC_KERNEL
/* Those are not defined in libkern */
#define PRIx64 "llx"
#define PRIx32 "x"
#define PRIx16 "hx"
#define PRIx8  "hhx"
#else
#include <inttypes.h>
#endif

#if  CCN_UNIT_SIZE == 8
#define CCPRIx_UNIT ".016" PRIx64
#elif  CCN_UNIT_SIZE == 4
#define CCPRIx_UNIT ".08" PRIx32
#else
#error invalid CCN_UNIT_SIZE
#endif

// ========================
// Print utilities for corecrypto
// ========================

#include <corecrypto/cc.h>

/* Print a byte array of arbitrary size */
void cc_print(const char *label, size_t count, const uint8_t *s);

#endif /* _CORECRYPTO_CCN_DEBUG_H_ */
