/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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
 * Copyright (c) 1993 Martin Birgmeier
 * All rights reserved.
 *
 * You may redistribute unmodified or modified versions of this source
 * code provided that the above copyright notice and this and the
 * following conditions are retained.
 *
 * This software is provided ``as is'', and comes with no warranties
 * of any kind. I shall in no event be liable for anything that happens
 * to anyone/anything when using this software.
 */

#include <sys/cdefs.h>
#if defined(LIBC_SCCS) && !defined(lint)
__RCSID("$NetBSD: lcong48.c,v 1.8 2005/06/12 05:21:28 lukem Exp $");
#endif /* LIBC_SCCS and not lint */

#include "namespace.h"

#include <assert.h>

#include "rand48.h"

#ifdef __weak_alias
__weak_alias(lcong48,_lcong48)
#endif

void
lcong48(unsigned short p[7])
{
	_DIAGASSERT(p != NULL);

	__rand48_seed[0] = p[0];
	__rand48_seed[1] = p[1];
	__rand48_seed[2] = p[2];
	__rand48_mult[0] = p[3];
	__rand48_mult[1] = p[4];
	__rand48_mult[2] = p[5];
	__rand48_add = p[6];
}

