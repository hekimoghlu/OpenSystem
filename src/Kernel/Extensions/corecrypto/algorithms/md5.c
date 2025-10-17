/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
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
//  pdcrypto_md5.c
//  pdcrypto
//
//  Created by rafirafi on 3/17/16.
//  Copyright (c) 2016 rafirafi. All rights reserved.
//

#include <stddef.h>
#include <corecrypto/ccmd5.h>

#include "pdcrypto_digest_final.h"

void pdcmd5_compress(ccdigest_state_t state, unsigned long nblocks, const void *data);

const uint32_t pdcmd5_initial_state[4] = {
    0x67452301UL, // A
    0xefcdab89UL, // B
    0x98badcfeUL, // C
    0x10325476UL  // D
};

#define pdcoid_md5_len  10

const struct ccdigest_info pdcmd5_di = {
    .output_size = CCMD5_OUTPUT_SIZE,
    .state_size = CCMD5_STATE_SIZE,
    .block_size = CCMD5_BLOCK_SIZE,
    .oid_size = pdcoid_md5_len,
    .oid = (unsigned char *)CC_DIGEST_OID_MD5,
    .initial_state = pdcmd5_initial_state,
    .compress = pdcmd5_compress,
    .final = pdcdigest_final_64le
};

