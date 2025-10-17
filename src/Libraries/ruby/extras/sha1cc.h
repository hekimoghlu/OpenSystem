/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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

#ifndef SHA1CC_H_INCLUDED
#define SHA1CC_H_INCLUDED

#include <CommonCrypto/CommonDigest.h>

#define SHA1_CTX	CC_SHA1_CTX

#define SHA1_BLOCK_LENGTH	CC_SHA1_BLOCK_BYTES
#define SHA1_DIGEST_LENGTH	CC_SHA1_DIGEST_LENGTH

#define SHA1_Init CC_SHA1_Init
#define SHA1_Update CC_SHA1_Update_Block
#define SHA1_Finish CC_SHA1_Finish

#define SHA1_STRIDE_SIZE	16384

void SHA1_Update(SHA1_CTX *context, const uint8_t *data, size_t len);
void SHA1_Finish(SHA1_CTX *ctx, char *buf);

#endif
