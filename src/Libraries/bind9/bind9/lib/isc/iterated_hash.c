/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
/* $Id: iterated_hash.c,v 1.6 2009/02/18 23:47:48 tbox Exp $ */

#include "config.h"

#include <stdio.h>

#include <isc/sha1.h>
#include <isc/iterated_hash.h>

int
isc_iterated_hash(unsigned char out[ISC_SHA1_DIGESTLENGTH],
		  unsigned int hashalg, int iterations,
		  const unsigned char *salt, int saltlength,
		  const unsigned char *in, int inlength)
{
	isc_sha1_t ctx;
	int n = 0;

	if (hashalg != 1)
		return (0);

	do {
		isc_sha1_init(&ctx);
		isc_sha1_update(&ctx, in, inlength);
		isc_sha1_update(&ctx, salt, saltlength);
		isc_sha1_final(&ctx, out);
		in = out;
		inlength = ISC_SHA1_DIGESTLENGTH;
	} while (n++ < iterations);

	return (ISC_SHA1_DIGESTLENGTH);
}
