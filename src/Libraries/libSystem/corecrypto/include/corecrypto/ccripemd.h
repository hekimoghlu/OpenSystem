/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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

#ifndef _CCRIPEMD_H_
#define _CCRIPEMD_H_

#include <corecrypto/ccdigest.h>

#define CCRMD_BLOCK_SIZE   64

#define CCRMD128_OUTPUT_SIZE  16
#define CCRMD160_OUTPUT_SIZE  20
#define CCRMD256_OUTPUT_SIZE  32
#define CCRMD320_OUTPUT_SIZE  40

#define CCRMD128_STATE_SIZE  16
#define CCRMD160_STATE_SIZE  20
#define CCRMD256_STATE_SIZE  32
#define CCRMD320_STATE_SIZE  40

extern const uint32_t ccrmd_initial_state[4];

extern const struct ccdigest_info ccrmd128_ltc_di;
extern const struct ccdigest_info ccrmd160_ltc_di;
extern const struct ccdigest_info ccrmd256_ltc_di;
extern const struct ccdigest_info ccrmd320_ltc_di;

#define ccrmd128_di ccrmd128_ltc_di
#define ccrmd160_di ccrmd160_ltc_di
#define ccrmd256_di ccrmd256_ltc_di
#define ccrmd320_di ccrmd320_ltc_di

#endif
