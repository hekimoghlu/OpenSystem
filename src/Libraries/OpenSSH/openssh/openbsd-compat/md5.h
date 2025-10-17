/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
 * This code implements the MD5 message-digest algorithm.
 * The algorithm is due to Ron Rivest.  This code was
 * written by Colin Plumb in 1993, no copyright is claimed.
 * This code is in the public domain; do with it what you wish.
 *
 * Equivalent code is available from RSA Data Security, Inc.
 * This code has been tested against that, and is equivalent,
 * except that you don't need to include two pages of legalese
 * with every copy.
 */

#ifndef _MD5_H_
#define _MD5_H_

#ifndef WITH_OPENSSL

#define	MD5_BLOCK_LENGTH		64
#define	MD5_DIGEST_LENGTH		16
#define	MD5_DIGEST_STRING_LENGTH	(MD5_DIGEST_LENGTH * 2 + 1)

typedef struct MD5Context {
	u_int32_t state[4];			/* state */
	u_int64_t count;			/* number of bits, mod 2^64 */
	u_int8_t buffer[MD5_BLOCK_LENGTH];	/* input buffer */
} MD5_CTX;

void	 MD5Init(MD5_CTX *);
void	 MD5Update(MD5_CTX *, const u_int8_t *, size_t)
		__attribute__((__bounded__(__string__,2,3)));
void	 MD5Pad(MD5_CTX *);
void	 MD5Final(u_int8_t [MD5_DIGEST_LENGTH], MD5_CTX *)
		__attribute__((__bounded__(__minbytes__,1,MD5_DIGEST_LENGTH)));
void	 MD5Transform(u_int32_t [4], const u_int8_t [MD5_BLOCK_LENGTH])
		__attribute__((__bounded__(__minbytes__,1,4)))
		__attribute__((__bounded__(__minbytes__,2,MD5_BLOCK_LENGTH)));
char	*MD5End(MD5_CTX *, char *)
		__attribute__((__bounded__(__minbytes__,2,MD5_DIGEST_STRING_LENGTH)));
char	*MD5File(const char *, char *)
		__attribute__((__bounded__(__minbytes__,2,MD5_DIGEST_STRING_LENGTH)));
char	*MD5FileChunk(const char *, char *, off_t, off_t)
		__attribute__((__bounded__(__minbytes__,2,MD5_DIGEST_STRING_LENGTH)));
char	*MD5Data(const u_int8_t *, size_t, char *)
		__attribute__((__bounded__(__string__,1,2)))
		__attribute__((__bounded__(__minbytes__,3,MD5_DIGEST_STRING_LENGTH)));

#endif /* !WITH_OPENSSL */

#endif /* _MD5_H_ */
