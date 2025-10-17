/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 19, 2025.
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
 * Copyright (c) 1983, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <net/if.h>

#include <err.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ifaddrs.h>

#include <net/if_dl.h>
#include <net/if_types.h>
#include <net/ethernet.h>

#include "ifconfig.h"

extern char *f_ether;

static struct ifreq link_ridreq;

static void
link_status(int s __unused, const struct ifaddrs *ifa)
{
	/* XXX no const 'cuz LLADDR is defined wrong */
	struct sockaddr_dl *sdl = (struct sockaddr_dl *) ifa->ifa_addr;

	if (sdl != NULL && sdl->sdl_alen > 0) {
		char *cp = (char *)LLADDR(sdl);
		int n = sdl->sdl_alen;
		char *format_char = ":";

		if (f_ether != NULL && strcmp(f_ether, "dash") == 0) {
			format_char = "-";
		}

		if (sdl->sdl_type == IFT_ETHER)
			printf ("\tether ");
		else
			printf ("\tlladdr ");
		while (--n >= 0)
			printf("%02x%s",*cp++ & 0xff, n>0? format_char : "");
		putchar('\n');
	}
}

static void
link_getaddr(const char *addr, int which)
{
	char *temp;
	struct sockaddr_dl sdl;
	struct sockaddr *sa = &link_ridreq.ifr_addr;
	size_t slen = strlen(addr);

	if (which != ADDR)
		errx(1, "can't set link-level netmask or broadcast");
	if ((temp = malloc(slen + 2)) == NULL)
		errx(1, "malloc failed");
	temp[0] = ':';
	strlcpy(temp + 1, addr, slen + 1);
	sdl.sdl_len = sizeof(sdl);
	link_addr(temp, &sdl);
	free(temp);
	if (sdl.sdl_alen > sizeof(sa->sa_data))
		errx(1, "malformed link-level address");
	sa->sa_family = AF_LINK;
	sa->sa_len = sdl.sdl_alen;
	bcopy(LLADDR(&sdl), sa->sa_data, sdl.sdl_alen);
}

static struct afswtch af_link = {
	.af_name	= "link",
	.af_af		= AF_LINK,
	.af_status	= link_status,
	.af_getaddr	= link_getaddr,
	.af_aifaddr	= SIOCSIFLLADDR,
	.af_addreq	= &link_ridreq,
};
static struct afswtch af_ether = {
	.af_name	= "ether",
	.af_af		= AF_LINK,
	.af_status	= link_status,
	.af_getaddr	= link_getaddr,
	.af_aifaddr	= SIOCSIFLLADDR,
	.af_addreq	= &link_ridreq,
};
static struct afswtch af_lladdr = {
	.af_name	= "lladdr",
	.af_af		= AF_LINK,
	.af_status	= link_status,
	.af_getaddr	= link_getaddr,
	.af_aifaddr	= SIOCSIFLLADDR,
	.af_addreq	= &link_ridreq,
};

static __constructor void
link_ctor(void)
{
	af_register(&af_link);
	af_register(&af_ether);
	af_register(&af_lladdr);
}
