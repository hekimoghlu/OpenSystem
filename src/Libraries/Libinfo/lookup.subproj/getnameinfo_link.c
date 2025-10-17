/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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
#include <stdint.h>
#include <stdio.h>
#include <netdb.h>
#include <net/if_types.h>
#include <net/if_dl.h>

static int
hexname(const uint8_t *cp, size_t len, char *host, size_t hostlen)
{
	int i, n;
	char *outp = host;

	*outp = '\0';
	for (i = 0; i < len; i++)
	{
		n = snprintf(outp, hostlen, "%s%02x", i ? ":" : "", cp[i]);
		if ((n < 0) || (n >= hostlen))
		{
			*host = '\0';
			return EAI_MEMORY;
		}

		outp += n;
		hostlen -= n;
	}

	return 0;
}

/*
 * getnameinfo_link():
 * Format a link-layer address into a printable format, paying attention to
 * the interface type.
 */
__private_extern__ int
getnameinfo_link(const struct sockaddr *sa, socklen_t salen, char *host, size_t hostlen, char *serv, size_t servlen, int flags)
{
	const struct sockaddr_dl *sdl = (const struct sockaddr_dl *)(const void *)sa;
	int n;

	if (serv != NULL && servlen > 0) *serv = '\0';

	if ((sdl->sdl_nlen == 0) && (sdl->sdl_alen == 0) && (sdl->sdl_slen == 0))
	{
		n = snprintf(host, hostlen, "link#%d", sdl->sdl_index);
		if (n > hostlen)
		{
			*host = '\0';
			return EAI_MEMORY;
		}

		return 0;
	}

	switch (sdl->sdl_type)
	{
			/*
			 * The following have zero-length addresses.
			 * IFT_ATM      (net/if_atmsubr.c)
			 * IFT_FAITH    (net/if_faith.c)
			 * IFT_GIF      (net/if_gif.c)
			 * IFT_LOOP     (net/if_loop.c)
			 * IFT_PPP      (net/if_ppp.c, net/if_spppsubr.c)
			 * IFT_SLIP     (net/if_sl.c, net/if_strip.c)
			 * IFT_STF      (net/if_stf.c)
			 * IFT_L2VLAN   (net/if_vlan.c)
			 * IFT_BRIDGE (net/if_bridge.h>
			 */
			/*
			 * The following use IPv4 addresses as link-layer addresses:
			 * IFT_OTHER    (net/if_gre.c)
			 * IFT_OTHER    (netinet/ip_ipip.c)
			 */
			/* default below is believed correct for all these. */
		case IFT_ARCNET:
		case IFT_ETHER:
		case IFT_FDDI:
		case IFT_HIPPI:
		case IFT_ISO88025:
		default:
			return hexname((uint8_t *)LLADDR(sdl), (size_t)sdl->sdl_alen, host, hostlen);
	}

	/* NOTREACHED */
	return EAI_FAIL;
}
