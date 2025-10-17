/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 31, 2022.
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
#if 0
#ifndef lint
static const char sccsid[] = "@(#)authenc.c	8.2 (Berkeley) 5/30/95";
#endif
#endif
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/contrib/telnet/telnetd/authenc.c,v 1.8 2003/05/04 02:54:49 obrien Exp $");

#if	defined(AUTHENTICATION) || defined(ENCRYPTION)
#include "telnetd.h"
#include <libtelnet/misc.h>

int
net_write(unsigned char *str, int len)
{
	if (nfrontp + len < netobuf + BUFSIZ) {
		output_datalen(str, len);
		return(len);
	}
	return(0);
}

void
net_encrypt(void)
{
#ifdef	ENCRYPTION
	char *s = (nclearto > nbackp) ? nclearto : nbackp;
	if (s < nfrontp && encrypt_output) {
		(*encrypt_output)((unsigned char *)s, nfrontp - s);
	}
	nclearto = nfrontp;
#endif /* ENCRYPTION */
}

int
telnet_spin(void)
{
	ttloop();
	return(0);
}

char *
telnet_getenv(char *val)
{
	return(getenv(val));
}

char *
telnet_gets(const char *prompt __unused, char *result __unused, int length __unused, int echo __unused)
{
	return(NULL);
}
#endif	/* ENCRYPTION */
#endif	/* AUTHENTICATION */
