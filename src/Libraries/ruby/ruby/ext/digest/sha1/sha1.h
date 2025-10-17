/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 22, 2023.
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
/*	$RoughId: sha1.h,v 1.3 2002/02/24 08:14:32 knu Exp $	*/
/*	$Id: sha1.h 52694 2015-11-21 04:35:57Z naruse $	*/

/*
 * SHA-1 in C
 * By Steve Reid <steve@edmweb.com>
 * 100% Public Domain
 */

#ifndef _SYS_SHA1_H_
#define	_SYS_SHA1_H_

#include "../defs.h"

typedef struct {
	uint32_t state[5];
	uint32_t count[2];
	uint8_t buffer[64];
} SHA1_CTX;

#ifdef RUBY
/* avoid name clash */
#define SHA1_Transform	rb_Digest_SHA1_Transform
#define SHA1_Init	rb_Digest_SHA1_Init
#define SHA1_Update	rb_Digest_SHA1_Update
#define SHA1_Finish	rb_Digest_SHA1_Finish
#endif

void	SHA1_Transform _((uint32_t state[5], const uint8_t buffer[64]));
int	SHA1_Init _((SHA1_CTX *context));
void	SHA1_Update _((SHA1_CTX *context, const uint8_t *data, size_t len));
int	SHA1_Finish _((SHA1_CTX *context, uint8_t digest[20]));

#define SHA1_BLOCK_LENGTH		64
#define SHA1_DIGEST_LENGTH		20
#define SHA1_DIGEST_STRING_LENGTH	(SHA1_DIGEST_LENGTH * 2 + 1)

#endif /* _SYS_SHA1_H_ */
