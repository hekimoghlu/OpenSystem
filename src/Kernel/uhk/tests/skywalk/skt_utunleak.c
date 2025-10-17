/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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
 * Attempts to allocate as many utuns as it can and then cleans
 * them up.  It does this twice because we originally had a leak
 * when we hit the limit the first time so the second time
 * would get EBUSY intead of ENOMEM
 */

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>

#include <uuid/uuid.h>
#include <sys/types.h>

#include <skywalk/os_skywalk.h>
#include <darwintest.h>

#include "skywalk_test_driver.h"
#include "skywalk_test_common.h"
#include "skywalk_test_utils.h"

static int
skt_utunleak_main(int argc, char *argv[])
{
	int nchannels = 500;
	int utuns[nchannels];
	int error;

	sktc_raise_file_limit(nchannels + 10);

	for (int i = 0; i < nchannels; i++) {
		utuns[i] = -1;
	}

	for (int i = 0; i < nchannels; i++) {
		utuns[i] = sktu_create_interface(SKTU_IFT_UTUN, SKTU_IFF_ENABLE_NETIF);
		if (utuns[i] == -1) {
			SKT_LOG("Expected: Failed on count %d errno %d\n", i + 1, errno);
			assert(errno != EBUSY);
			assert(errno == ENOMEM);
			break;
		}
	}
	for (int i = 0; i < nchannels; i++) {
		if (utuns[i] != -1) {
			error = close(utuns[i]);
			SKTC_ASSERT_ERR(!error);
			utuns[i] = -1;
		}
	}

	/* Now try it a second time and verify it works the same */

	for (int i = 0; i < nchannels; i++) {
		utuns[i] = -1;
	}

	for (int i = 0; i < nchannels; i++) {
		utuns[i] = sktu_create_interface(SKTU_IFT_UTUN, SKTU_IFF_ENABLE_NETIF);
		if (utuns[i] == -1) {
			SKT_LOG("Expected: Failed on count %d errno %d\n", i + 1, errno);
			assert(errno != EBUSY);
			assert(errno == ENOMEM);
			break;
		}
	}

	for (int i = 0; i < nchannels; i++) {
		if (utuns[i] != -1) {
			error = close(utuns[i]);
			SKTC_ASSERT_ERR(!error);
			utuns[i] = -1;
		}
	}

	return 0;
}

struct skywalk_test skt_utunleak = {
	"utunleak", "allocate utuns until failure to reproduce a leak",
	SK_FEATURE_SKYWALK | SK_FEATURE_NEXUS_KERNEL_PIPE,
	skt_utunleak_main,
};
