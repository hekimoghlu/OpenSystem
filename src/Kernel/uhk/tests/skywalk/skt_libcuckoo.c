/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include <uuid/uuid.h>

#include "skywalk_test_driver.h"
#include "skywalk_test_common.h"
#include "skywalk_test_utils.h"

static int sktc_libcuckoo_test_was_enabled;

void
sktc_libcuckoo_init(void)
{
	size_t len = sizeof(sktc_libcuckoo_test_was_enabled);
	int enabled = 1;

	sysctlbyname("kern.skywalk.libcuckoo.test", &sktc_libcuckoo_test_was_enabled,
	    &len, &enabled, sizeof(enabled));
}

void
sktc_libcuckoo_fini(void)
{
	sysctlbyname("kern.skywalk.libcuckoo.test", NULL, 0,
	    &sktc_libcuckoo_test_was_enabled,
	    sizeof(sktc_libcuckoo_test_was_enabled));
}

static int
skt_libcuckoo_main(int argc, char *argv[])
{
#pragma unused(argc, argv)
	/*
	 * A failure for this test is indicated by either a panic or
	 * a hang; we rely on some external mechanism to detect the
	 * latter and take the appropriate actions.
	 */
	return 0;
}

struct skywalk_test skt_libcuckoo = {
	"libcuckoo", "Cuckoo hashtable library basic and advanced tests",
	SK_FEATURE_SKYWALK | SK_FEATURE_DEV_OR_DEBUG,
	skt_libcuckoo_main, { NULL },
	sktc_libcuckoo_init, sktc_libcuckoo_fini,
};
