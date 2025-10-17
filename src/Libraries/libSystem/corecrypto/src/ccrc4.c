/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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

#include <corecrypto/ccrc4.h>
#include <corecrypto/ccstubs.h>
#include <stdio.h>

const struct ccrc4_info *ccrc4(void) {
	CC_STUB(NULL);
}

int ccrc4_test(const struct ccrc4_info *rc4, const struct ccrc4_vector *v) {
	CC_STUB(0);
}

typedef uint32_t eay_RC4_INT;

typedef struct eay_rc4_key_st
{
	eay_RC4_INT x,y;
	eay_RC4_INT data[256];
} eay_RC4_KEY;

static void eay_RC4(ccrc4_ctx *skey, unsigned long len, const void *in, void *out) {
	CC_STUB_VOID();
}

static void eay_RC4_set_key(ccrc4_ctx *skey, size_t keylen, const void *keydata) {
	CC_STUB_VOID();
}

const struct ccrc4_info ccrc4_eay = {
    .size = sizeof(eay_RC4_KEY),
    .init = eay_RC4_set_key,
    .crypt = eay_RC4,
};
