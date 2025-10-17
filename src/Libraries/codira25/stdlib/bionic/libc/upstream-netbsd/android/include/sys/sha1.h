/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 20, 2025.
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
 * SHA-1 in C
 * By Steve Reid <steve@edmweb.com>
 * 100% Public Domain
 */

#ifndef _SYS_SHA1_H_
#define	_SYS_SHA1_H_

#include <sys/cdefs.h>
#include <sys/types.h>

#define SHA1_DIGEST_LENGTH		20
#define SHA1_DIGEST_STRING_LENGTH	41

typedef struct {
	uint32_t state[5];
	uint32_t count[2];
	u_char buffer[64];
} SHA1_CTX;

__BEGIN_DECLS
void	SHA1Transform(uint32_t[5], const u_char[64]);
void	SHA1Init(SHA1_CTX *);
void	SHA1Update(SHA1_CTX *, const u_char *, u_int);
void	SHA1Final(u_char[SHA1_DIGEST_LENGTH], SHA1_CTX *);
__END_DECLS

#endif /* _SYS_SHA1_H_ */
