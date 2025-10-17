/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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
#ifndef _CRYPTO_SHA1_H_
#define _CRYPTO_SHA1_H_

#if KERNEL
#include <sys/types.h>
#else /* !KERNEL */
#include <machine/types.h>
#endif /* KERNEL */

#ifdef  __cplusplus
extern "C" {
#endif

#define SHA_DIGEST_LENGTH       20
#define SHA1_RESULTLEN          SHA_DIGEST_LENGTH

typedef struct sha1_ctxt {
	union {
		u_int8_t        b8[20];
		u_int32_t       b32[5]; /* state (ABCDE) */
	} h;
	union {
		u_int8_t        b8[8];
		u_int32_t       b32[2];
		u_int64_t       b64[1]; /* # of bits, modulo 2^64 (msb first) */
	} c;
	union {
		u_int8_t        b8[64];
		u_int32_t       b32[16]; /* input buffer */
	} m;
	u_int8_t        count;          /* unused; for compatibility only */
} SHA1_CTX;

/* For compatibility with the other SHA-1 implementation. */
#define sha1_init(c)            SHA1Init(c)
#define sha1_loop(c, b, l)      SHA1Update(c, b, l)
#define sha1_result(c, b)       SHA1Final(b, c)

extern void SHA1Init(SHA1_CTX *);
extern void SHA1Update(SHA1_CTX *, const void *, size_t);
extern void SHA1Final(void *, SHA1_CTX *);

#ifdef  __cplusplus
}
#endif

#endif /*_CRYPTO_SHA1_H_*/
