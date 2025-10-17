/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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
#include <unistd.h>
#include <stdint.h>
#include <sys/time.h>
#include <mach/mach_time.h>
#include <sys/codesign.h>
#include <mach/mach.h>
#include <darwintest.h>
#include <stdlib.h>
#include "cs_helpers.h"


int
remove_platform_binary(void)
{
	int ret;
	uint32_t my_csflags;

	T_QUIET; T_ASSERT_POSIX_ZERO(csops(getpid(), CS_OPS_STATUS, &my_csflags, sizeof(my_csflags)), NULL);

	if (!(my_csflags & CS_PLATFORM_BINARY)) {
		return 0;
	}

	ret = csops(getpid(), CS_OPS_CLEARPLATFORM, NULL, 0);
	if (ret) {
		switch (errno) {
		case ENOTSUP:
			T_LOG("clearing platform binary not supported, skipping test");
			return -1;
		default:
			T_LOG("csops failed with flag CS_OPS_CLEARPLATFORM");
			return -1;
		}
	}

	my_csflags = 0;
	T_QUIET; T_ASSERT_POSIX_ZERO(csops(getpid(), CS_OPS_STATUS, &my_csflags, sizeof(my_csflags)), NULL);

	if (my_csflags & CS_PLATFORM_BINARY) {
		T_LOG("platform binary flag still set");
		return -1;
	}

	return 0;
}
