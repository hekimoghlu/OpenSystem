/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 2, 2023.
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
static const char sccsid[] = "@(#)authenc.c	8.1 (Berkeley) 6/6/93";
#endif
#endif
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/contrib/telnet/telnet/authenc.c,v 1.6 2003/05/04 02:54:48 obrien Exp $");

#ifdef	AUTHENTICATION
#include <sys/types.h>
#include <arpa/telnet.h>
#include <pwd.h>
#include <unistd.h>
#include <libtelnet/encrypt.h>
#include <libtelnet/misc.h>

#include "general.h"
#include "ring.h"
#include "externs.h"
#include "defines.h"
#include "types.h"

int
net_write(unsigned char *str, int len)
{
	if (NETROOM() > len) {
		ring_supply_data(&netoring, str, len);
		if (str[0] == IAC && str[1] == SE)
			printsub('>', &str[2], len-2);
		return(len);
	}
	return(0);
}

void
net_encrypt(void)
{
#ifdef	ENCRYPTION
	if (encrypt_output)
		ring_encrypt(&netoring, encrypt_output);
	else
		ring_clearto(&netoring);
#endif	/* ENCRYPTION */
}

int
telnet_spin(void)
{
	return(-1);
}

char *
telnet_getenv(char *val)
{
	return((char *)env_getvalue((unsigned char *)val));
}

char *
telnet_gets(const char *prom, char *result, int length, int echo)
{
	extern int globalmode;
	int om = globalmode;
	char *res;

	TerminalNewMode(-1);
	if (echo) {
		printf("%s", prom);
		res = fgets(result, length, stdin);
	} else if ((res = getpass(prom))) {
		strncpy(result, res, length);
		res = result;
	}
	TerminalNewMode(om);
	return(res);
}
#endif	/* AUTHENTICATION */
