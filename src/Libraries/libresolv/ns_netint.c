/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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
#ifndef __APPLE__
#ifndef lint
static const char rcsid[] = "$Id: ns_netint.c,v 1.3 2005/04/27 04:56:40 sra Exp $";
#endif
#endif /* ! __APPLE__ */
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

/* Import. */

#include "port_before.h"

#include <arpa/nameser.h>

#include "port_after.h"

/* Public. */

u_int
ns_get16(const u_char *src) {
	u_int dst;

	NS_GET16(dst, src);
	return (dst);
}

u_long
ns_get32(const u_char *src) {
	u_long dst;

	NS_GET32(dst, src);
	return (dst);
}

void
ns_put16(u_int src, u_char *dst) {
	NS_PUT16(src, dst);
}

void
ns_put32(u_long src, u_char *dst) {
	NS_PUT32(src, dst);
}

/*! \file */
