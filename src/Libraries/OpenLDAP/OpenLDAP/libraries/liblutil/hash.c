/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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
/* This work is part of OpenLDAP Software <http://www.openldap.org/>.
 *
 * Copyright 2000-2011 The OpenLDAP Foundation.
 * Portions Copyright 2000-2003 Kurt D. Zeilenga.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted only as authorized by the OpenLDAP
 * Public License.
 *
 * A copy of this license is available in the file LICENSE in the
 * top-level directory of the distribution or, alternatively, at
 * <http://www.OpenLDAP.org/license.html>.
 */

/* This implements the Fowler / Noll / Vo (FNV-1) hash algorithm.
 * A summary of the algorithm can be found at:
 *   http://www.isthe.com/chongo/tech/comp/fnv/index.html
 */

#include "portable.h"

#include <lutil_hash.h>

/* offset and prime for 32-bit FNV-1 */
#define HASH_OFFSET	0x811c9dc5U
#define HASH_PRIME	16777619


/*
 * Initialize context
 */
void
lutil_HASHInit( struct lutil_HASHContext *ctx )
{
	ctx->hash = HASH_OFFSET;
}

/*
 * Update hash
 */
void
lutil_HASHUpdate(
    struct lutil_HASHContext	*ctx,
    const unsigned char		*buf,
    ber_len_t		len )
{
	const unsigned char *p, *e;
	ber_uint_t h;

	p = buf;
	e = &buf[len];

	h = ctx->hash;

	while( p < e ) {
		h *= HASH_PRIME;
		h ^= *p++;
	}

	ctx->hash = h;
}

/*
 * Save hash
 */
void
lutil_HASHFinal( unsigned char *digest, struct lutil_HASHContext *ctx )
{
	ber_uint_t h = ctx->hash;

	digest[0] = h & 0xffU;
	digest[1] = (h>>8) & 0xffU;
	digest[2] = (h>>16) & 0xffU;
	digest[3] = (h>>24) & 0xffU;
}
