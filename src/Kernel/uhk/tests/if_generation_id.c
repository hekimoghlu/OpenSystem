/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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
#include <darwintest.h>

#include <sys/ioctl.h>

#include <net/if.h>
#include <net/if_arp.h>
#include <net/if_fake_var.h>

#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "net_test_lib.h"

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.net"),
	T_META_ASROOT(true),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("networking"),
	T_META_CHECK_LEAKS(false));

static char ifname1[IF_NAMESIZE];

static void
cleanup(void)
{
	if (ifname1[0] != '\0') {
		(void)ifnet_destroy(ifname1, false);
		T_LOG("ifnet_destroy %s", ifname1);
	}
}

T_DECL(if_creation_generation_id, "network interface creation generation id")
{
	int     error;
	int     s = inet_dgram_socket_get();

	T_ATEND(cleanup);

#ifdef SIOCGIFGENERATIONID
	strlcpy(ifname1, FETH_NAME, sizeof(ifname1));
	error = ifnet_create_2(ifname1, sizeof(ifname1));
	if (error != 0) {
		ifname1[0] = '\0';
		T_ASSERT_POSIX_SUCCESS(error, "ifnet_create_2");
	}
	T_LOG("created %s", ifname1);

	struct ifreq ifr = {};

	strlcpy(ifr.ifr_name, ifname1, sizeof(ifr.ifr_name));

	T_ASSERT_POSIX_SUCCESS(ioctl(s, SIOCGIFGENERATIONID, &ifr), NULL);

	uint64_t if_generation_id = ifr.ifr_creation_generation_id;
	T_LOG("interface creation generation id: %llu", if_generation_id);

	(void)ifnet_destroy(ifname1, true);
	T_LOG("destroyed %s", ifname1);

	/* ifnet_create() will retry if creating fails due to EBUSY */
	T_ASSERT_POSIX_SUCCESS(ifnet_create(ifname1), NULL);

	T_LOG("re-created %s", ifname1);

	T_ASSERT_POSIX_SUCCESS(ioctl(s, SIOCGIFGENERATIONID, &ifr), NULL);

	T_LOG("interface creation generation id: %llu", ifr.ifr_creation_generation_id);

	T_ASSERT_NE_ULLONG(if_generation_id, ifr.ifr_creation_generation_id,
	    "interface generation id are different");
#else
	T_SKIP("SIOCGIFGENERATIONID does not exist");
#endif
}
