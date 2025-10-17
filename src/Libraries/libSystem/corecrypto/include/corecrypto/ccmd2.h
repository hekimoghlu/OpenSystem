/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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
#ifndef _CORECRYPTO_CCMD2_H
#define _CORECRYPTO_CCMD2_H

#include <corecrypto/ccdigest.h>

#define CCMD2_BLOCK_SIZE   16
#define CCMD2_OUTPUT_SIZE  16
#define CCMD2_STATE_SIZE   64

extern const uint32_t ccmd2_initial_state[16];

#define ccmd2_di ccmd2_ltc_di
extern const struct ccdigest_info ccmd2_ltc_di;

void ccmd2_final(const struct ccdigest_info* di, ccdigest_ctx_t, unsigned char* digest);

#endif

