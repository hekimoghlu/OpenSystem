/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <darwintest.h>
#include "skywalk_test_driver.h"
#include "skywalk_test_common.h"

/****************************************************************/

static int
skt_mp100noop_main(int argc, char *argv[])
{
	char buf[1] = { 0 };
	assert(!strcmp(argv[3], "--child"));
	ssize_t ret;

	if ((ret = write(MPTEST_SEQ_FILENO, buf, sizeof(buf))) == -1) {
		SKT_LOG("write fail: %s", strerror(errno));
		return 1;
	}
	assert(ret == 1);

	/* Wait for go signal */
	if ((ret = read(MPTEST_SEQ_FILENO, buf, sizeof(buf))) == -1) {
		SKT_LOG("read fail: %s", strerror(errno));
		return 1;
	}
	assert(ret == 1);

	return 0;
}

struct skywalk_mptest skt_mp100noop = {
	"mp100noop", "test just returns true from 100 children", 0, 100, skt_mp100noop_main,
};
