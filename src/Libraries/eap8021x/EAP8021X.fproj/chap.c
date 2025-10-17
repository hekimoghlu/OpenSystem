/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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
/* 
 * Modification History
 *
 * December 10, 2001	Dieter Siegmund (dieter@apple.com)
 * - created
 */

/*
 * Function: CHAP_md5_hash
 * Purpose:
 *   Compute the CHAP MD5 hash using the method described in
 *   RFC 1994.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <CommonCrypto/CommonDigest.h>
#include "chap.h"

void
chap_md5(uint8_t identifier, const uint8_t * password, int password_len,
	 const uint8_t * challenge, int challenge_len,
	 uint8_t * hash)
{
    CC_MD5_CTX			ctx;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
    /* MD5 hash over the identifier + password + challenge */
    CC_MD5_Init(&ctx);
    CC_MD5_Update(&ctx, &identifier, sizeof(identifier));
    CC_MD5_Update(&ctx, password, password_len);
    CC_MD5_Update(&ctx, challenge, challenge_len);
    CC_MD5_Final(hash, &ctx);
#pragma GCC diagnostic pop
    return;
}
