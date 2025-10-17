/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 27, 2024.
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
/* This test attaches a flow switch to itself.
 */

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <darwintest.h>

#include "skywalk_test_common.h"
#include "skywalk_test_driver.h"
#include "skywalk_test_utils.h"

static void
skt_fswloop_common(const char *name)
{
	uuid_t attach;
	int error;
	struct sktc_nexus_attr attr = SKTC_NEXUS_ATTR_INIT();

	strncpy((char *)attr.name, name, sizeof(nexus_name_t) - 1);
	attr.type = NEXUS_TYPE_FLOW_SWITCH;
	attr.anonymous = 1;

	sktc_setup_nexus(&attr);

	error = __os_nexus_ifattach(sktc_nexus_controller, sktc_instance_uuid,
	    NULL, sktc_instance_uuid, false, &attach);
	SKTC_ASSERT_ERR(error == -1);
	SKTC_ASSERT_ERR(errno == EINVAL);
}

static int
skt_fswloopfsw_main(int argc, char *argv[])
{
	skt_fswloop_common("skywalk_test_fswloop");
	return 0;
}

struct skywalk_test skt_fswloopfsw = {
	"fswloopfsw", "create a flow-switch and attach it to itself",
	SK_FEATURE_SKYWALK | SK_FEATURE_NEXUS_FLOWSWITCH,
	skt_fswloopfsw_main,
};

void
skt_fswloop2_common(boolean_t add_netif)
{
	int error;
	nexus_controller_t ncd;
	uuid_t provider0, instance0, provider1, instance1, attach;
	uuid_t feth0_attach, feth1_attach;
	uuid_t netif_provider, netif0_instance, netif1_instance;
	struct sktc_nexus_attr attr = SKTC_NEXUS_ATTR_INIT();

	ncd = os_nexus_controller_create();
	assert(ncd);

	strncpy((char *)attr.name, "skt_fswloop2_zero",
	    sizeof(nexus_name_t) - 1);
	attr.type = NEXUS_TYPE_FLOW_SWITCH;
	attr.anonymous = 1;
	sktc_build_nexus(ncd, &attr, &provider0, &instance0);

	strncpy((char *)attr.name, "skt_fswloop2_one",
	    sizeof(nexus_name_t) - 1);
	sktc_build_nexus(ncd, &attr, &provider1, &instance1);

	if (add_netif) {
		strncpy((char *)attr.name, "skt_netif_feth",
		    sizeof(nexus_name_t) - 1);
		attr.type = NEXUS_TYPE_NET_IF;
		sktc_build_nexus(ncd, &attr, &netif_provider, &netif0_instance);

		error = __os_nexus_ifattach(ncd, netif0_instance, FETH0_NAME,
		    NULL, false, &feth0_attach);
		SKTC_ASSERT_ERR(!error);

		error = os_nexus_controller_alloc_provider_instance(ncd,
		    netif_provider, &netif1_instance);
		SKTC_ASSERT_ERR(!error);

		error = __os_nexus_ifattach(ncd, netif1_instance, FETH1_NAME,
		    NULL, false, &feth1_attach);
		SKTC_ASSERT_ERR(!error);
	}

	error = __os_nexus_ifattach(ncd, instance1, NULL, instance0, false, &attach);
	//T_LOG("%s:%d error %d errno %d\n", __func__, __LINE__, error, errno);
	/* Can't attach a flowswitch to anything */
	SKTC_ASSERT_ERR(error == -1);
	SKTC_ASSERT_ERR(errno == EINVAL);

	/* Now also try to attach them the other way around */
	error = __os_nexus_ifattach(ncd, instance0, NULL, instance1, false, &attach);
	//T_LOG("%s:%d error %d errno %d\n", __func__, __LINE__, error, errno);
	/* Can't attach a flowswitch to anything */
	SKTC_ASSERT_ERR(error == -1);
	SKTC_ASSERT_ERR(errno == EINVAL);
}

int
skt_fswloop2ff_main(int argc, char *argv[])
{
	skt_fswloop2_common(false);
	return 0;
}

struct skywalk_test skt_fswloop2ff = {
	"fswloop2mm", "attach a flowswitch to a flowswitch without any netif",
	SK_FEATURE_SKYWALK | SK_FEATURE_NEXUS_FLOWSWITCH,
	skt_fswloop2ff_main,
};

int
skt_fswloop2nff_main(int argc, char *argv[])
{
	skt_fswloop2_common(true);
	return 0;
}

struct skywalk_test skt_fswloop2nmm = {
	"fswloop2nmm", "attach a flowswitch to a flowswitch and back to itself",
	SK_FEATURE_SKYWALK | SK_FEATURE_NEXUS_NETIF | SK_FEATURE_NEXUS_FLOWSWITCH,
	skt_fswloop2nff_main, { NULL },
	sktc_ifnet_feth0_1_create, sktc_ifnet_feth0_1_destroy,
};
