/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
 * MD5.H - header file for MD5.C
 */

/*
 * Copyright (C) 1991-2, RSA Data Security, Inc. Created 1991. All
 * rights reserved.
 *
 * License to copy and use this software is granted provided that it
 * is identified as the "RSA Data Security, Inc. MD5 Message-Digest
 * Algorithm" in all material mentioning or referencing this software
 * or this function.
 *
 * License is also granted to make and use derivative works provided
 * that such works are identified as "derived from the RSA Data
 * Security, Inc. MD5 Message-Digest Algorithm" in all material
 * mentioning or referencing the derived work.
 *
 * RSA Data Security, Inc. makes no representations concerning either
 * the merchantability of this software or the suitability of this
 * software for any particular purpose. It is provided "as is"
 * without express or implied warranty of any kind.
 *
 * These notices must be retained in any copies of any part of this
 * documentation and/or software.
 */

#ifndef _CRYPTO_MD5_H_
#define _CRYPTO_MD5_H_

#if KERNEL
#include <sys/types.h>
#else /* !KERNEL */
#include <machine/types.h>
#endif /* KERNEL */
#include <sys/cdefs.h>

__BEGIN_DECLS

#define MD5_DIGEST_LENGTH       16

/* MD5 context. */
typedef struct {
	u_int32_t state[4];     /* state (ABCD) */
	u_int32_t count[2];     /* number of bits, modulo 2^64 (lsb first) */
	unsigned char buffer[64];       /* input buffer */
} MD5_CTX;

extern void MD5Init(MD5_CTX *);
extern void MD5Update(MD5_CTX *, const void *, unsigned int);
extern void MD5Final(unsigned char [MD5_DIGEST_LENGTH], MD5_CTX *);

__END_DECLS

#endif /* _CRYPTO_MD5_H_ */
