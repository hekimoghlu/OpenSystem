/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 16, 2024.
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
 * Copyright (c) 2020 Mark Kettenis <kettenis@openbsd.org>
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <sys/param.h>
#include <sys/device.h>
#include <sys/reboot.h>
#include <sys/systm.h>

#if defined(NFSCLIENT)
#include <net/if.h>
#include <net/if_types.h>
#include <netinet/in.h>
#include <netinet/if_ether.h>
#endif

void
cpu_configure(void)
{
	splhigh();

	softintr_init();
	config_rootfound("mainbus", NULL);

	cold = 0;
	spl0();
}

void
diskconf(void)
{
#if defined(NFSCLIENT)
	extern uint8_t *bootmac;
	dev_t tmpdev = NODEV;
#endif
	struct device *bootdv = NULL;
	int part = 0;

#if defined(NFSCLIENT)
	if (bootmac) {
		struct ifnet *ifp;

		TAILQ_FOREACH(ifp, &ifnetlist, if_list) {
			if (ifp->if_type == IFT_ETHER &&
			    memcmp(bootmac, ((struct arpcom *)ifp)->ac_enaddr,
			    ETHER_ADDR_LEN) == 0)
				break;
		}
		if (ifp)
			bootdv = parsedisk(ifp->if_xname, strlen(ifp->if_xname),
			    0, &tmpdev);
	}
#endif

	setroot(bootdv, part, RB_USERREQ);
}

void
device_register(struct device *dev, void *aux)
{
}

const struct nam2blk nam2blk[] = {
	{ "vnd",	1 },
	{ "rd",		2 },
	{ "sd",		3 },
	{ "cd",		4 },
	{ "wd",		5 },
	{ NULL,		-1 }
};

