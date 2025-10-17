/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 17, 2022.
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

#ifndef MD5CC_H_INCLUDED
#define MD5CC_H_INCLUDED

#include <stddef.h>
#include <CommonCrypto/CommonDigest.h>

#define MD5_CTX		CC_MD5_CTX

#define MD5_DIGEST_LENGTH	CC_MD5_DIGEST_LENGTH
#define MD5_BLOCK_LENGTH	CC_MD5_BLOCK_BYTES

#define MD5_Init CC_MD5_Init
#define MD5_Update CC_MD5_Update
#define MD5_Finish CC_MD5_Finish

void MD5_Finish(MD5_CTX *pctx, unsigned char *digest);

#endif
