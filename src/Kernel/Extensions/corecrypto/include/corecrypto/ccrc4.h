/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 11, 2025.
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
#ifndef _CORECRYPTO_CCRC4_H_
#define _CORECRYPTO_CCRC4_H_

#include <corecrypto/ccmode.h>

cc_aligned_struct(16) ccrc4_ctx;

/* Declare a rc4 key named _name_.  Pass the size field of a struct ccmode_ecb
 for _size_. */
#define ccrc4_ctx_decl(_size_, _name_) cc_ctx_decl(ccrc4_ctx, _size_, _name_)
#define ccrc4_ctx_clear(_size_, _name_) cc_clear(_size_, _name_)

struct ccrc4_info {
    size_t size;        /* first argument to ccrc4_ctx_decl(). */
    void (*init)(ccrc4_ctx *ctx, size_t key_len, const void *key);
    void (*crypt)(ccrc4_ctx *ctx, size_t nbytes, const void *in, void *out);
};

const struct ccrc4_info *ccrc4(void);

extern const struct ccrc4_info ccrc4_eay;

#endif /* _CORECRYPTO_CCRC4_H_ */
