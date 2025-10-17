/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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
/* <rdar://problem/58673168> Restricted port used by non-entitled process */

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include "skywalk_test_driver.h"
#include "skywalk_test_utils.h"
#include "skywalk_test_common.h"

static int
skt_reserve_restricted_port()
{
	int error;
	int old_first, old_last;
	int restricted_port = 55555;
	size_t size;

	size = sizeof(old_first);
	error = sysctlbyname("net.inet.ip.portrange.first", &old_first, &size, &restricted_port, sizeof(restricted_port));
	SKTC_ASSERT_ERR(!error);
	assert(size == sizeof(old_first));

	size = sizeof(old_last);
	error = sysctlbyname("net.inet.ip.portrange.last", &old_last, &size, &restricted_port, sizeof(restricted_port));
	SKTC_ASSERT_ERR(!error);
	assert(size == sizeof(old_last));

	struct sktc_nexus_handles handles;
	sktc_create_flowswitch(&handles, 0);

	uuid_t flow;

	/* try reserve one of the restricted ephemeral ports */
	uuid_generate_random(flow);
	error = sktc_bind_tcp4_flow(handles.controller, handles.fsw_nx_uuid,
	    0, NEXUS_PORT_FLOW_SWITCH_CLIENT, flow);
	SKTC_ASSERT_ERR(error == -1);
	SKTC_ASSERT_ERR(errno == EADDRNOTAVAIL);
	uuid_clear(flow);

	sktc_cleanup_flowswitch(&handles);

	size = sizeof(old_first);
	error = sysctlbyname("net.inet.ip.portrange.first", NULL, NULL, &old_first, size);
	SKTC_ASSERT_ERR(!error);
	assert(size == sizeof(old_first));

	size = sizeof(old_last);
	error = sysctlbyname("net.inet.ip.portrange.last", NULL, NULL, &old_last, size);
	SKTC_ASSERT_ERR(!error);
	assert(size == sizeof(old_last));
	return 0;
}

int
skt_reserve_restricted_port_main(int argc, char *argv[])
{
	return skt_reserve_restricted_port();
}


struct skywalk_test skt_restricted_port = {
	"restricted_port", "test reserve a restricted ephemeral port",
	SK_FEATURE_SKYWALK | SK_FEATURE_NEXUS_NETIF | SK_FEATURE_NEXUS_FLOWSWITCH | SK_FEATURE_NETNS,
	skt_reserve_restricted_port_main, { NULL }, sktc_ifnet_feth0_create, sktc_ifnet_feth0_destroy,
};
