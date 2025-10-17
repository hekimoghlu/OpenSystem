/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 22, 2025.
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
 * nexus.c
 * - report information about attached nexus
 */

/*
 * Modification History:
 *
 * April 10, 2017	Dieter Siegmund (dieter@apple.com)
 * - created
 */

#include <sys/param.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

#include <stdlib.h>
#include <unistd.h>

#include <net/ethernet.h>
#include <net/if.h>
#include <net/if_var.h>
#include <net/if_fake_var.h>

#include <net/route.h>

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <err.h>
#include <errno.h>

#include "ifconfig.h"

static void
nexus_status(int s)
{
	struct if_nexusreq	ifnr;
	uuid_string_t		flowswitch;
	uuid_string_t		netif;

	if (!verbose) {
		return;
	}
	bzero((char *)&ifnr, sizeof(ifnr));
	strlcpy(ifnr.ifnr_name, ifr.ifr_name, sizeof(ifnr.ifnr_name));
	if (ioctl(s, SIOCGIFNEXUS, &ifnr) < 0) {
		return;
	}
	if (uuid_is_null(ifnr.ifnr_netif)) {
		/* technically, this shouldn't happen */
		return;
	}
	uuid_unparse_upper(ifnr.ifnr_netif, netif);
	printf("\tnetif: %s\n", netif);
	if (uuid_is_null(ifnr.ifnr_flowswitch) == 0) {
		uuid_unparse_upper(ifnr.ifnr_flowswitch, flowswitch);
		printf("\tflowswitch: %s\n", flowswitch);
	}
	return;
}

static struct afswtch af_fake = {
	.af_name	= "af_fake",
	.af_af		= AF_UNSPEC,
	.af_other_status = nexus_status,
};

static __constructor void
fake_ctor(void)
{
	af_register(&af_fake);
}

