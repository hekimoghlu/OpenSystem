/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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
#include <assert.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <darwintest.h>
#include "skywalk_test_driver.h"
#include "skywalk_test_common.h"

/****************************************************************/

static int
skt_features_main(int argc, char *argv[])
{
	size_t len;
	uint64_t features;
	int error;

	features = 0;
	len = sizeof(features);
	error = sysctlbyname("kern.skywalk.features", &features, &len, NULL, 0);
	SKTC_ASSERT_ERR(error == 0);
	assert(len == sizeof(features));

	T_LOG("features = 0x%016"PRIx64, features);

	assert(features & SK_FEATURE_SKYWALK);
	assert(features & SK_FEATURE_NETNS);
	assert(features & SK_FEATURE_NEXUS_USER_PIPE);
	assert(features & SK_FEATURE_NEXUS_KERNEL_PIPE);
	assert(features & SK_FEATURE_NEXUS_MONITOR);
	assert(features & SK_FEATURE_NEXUS_FLOWSWITCH);
	assert(features & SK_FEATURE_NEXUS_NETIF);

	if (features & (SK_FEATURE_DEVELOPMENT | SK_FEATURE_DEBUG)) {
		assert(features & SK_FEATURE_NEXUS_KERNEL_PIPE_LOOPBACK);
		assert(features & SK_FEATURE_DEV_OR_DEBUG);
	} else {
		assert(!(features & SK_FEATURE_NEXUS_KERNEL_PIPE_LOOPBACK));
		assert(!(features & SK_FEATURE_DEV_OR_DEBUG));
	}

	return 0;
}

struct skywalk_test skt_features = {
	"features", "verifies skywalk features match kernel config", 0, skt_features_main,
};
