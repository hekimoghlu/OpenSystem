/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 1, 2023.
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
#include "telnetd.h"

RCSID("$Id$");

#ifdef AUTHENTICATION

int
telnet_net_write(unsigned char *str, int len)
{
    if (nfrontp + len < netobuf + BUFSIZ) {
	memmove(nfrontp, str, len);
	nfrontp += len;
	return(len);
    }
    return(0);
}

void
net_encrypt(void)
{
#ifdef ENCRYPTION
    char *s = (nclearto > nbackp) ? nclearto : nbackp;
    if (s < nfrontp && encrypt_output) {
	(*encrypt_output)((unsigned char *)s, nfrontp - s);
    }
    nclearto = nfrontp;
#endif
}

int
telnet_spin(void)
{
    return ttloop();
}

char *
telnet_getenv(const char *val)
{
    return(getenv(val));
}

char *
telnet_gets(char *prompt, char *result, int length, int echo)
{
    return NULL;
}
#endif
