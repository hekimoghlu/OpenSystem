/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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
#include "skywalk_test_driver.h"
#include "skywalk_test_common.h"

/****************************************************************/

static int
skt_noop_main(int argc, char *argv[])
{
	return 0;
}

struct skywalk_test skt_noop = {
	"noop", "test just returns true", 0, skt_noop_main,
};

/****************************************************************/

static int
skt_crash_main(int argc, char *argv[])
{
	*(volatile int *)0 = 1; // Crash
	return 1;
}

struct skywalk_test skt_crash = {
	"crash", "test expects a segfault",
	0, skt_crash_main, { NULL }, NULL, NULL, 0xb100001, 0,
};

/****************************************************************/

static int
skt_assert_main(int argc, char *argv[])
{
	assert(0);
	return 1;
}

struct skywalk_test skt_assert = {
	"assert", "test verifies that assert catches failure",
	0, skt_assert_main, { NULL }, NULL, NULL, 0x6000000, 0,
};

/****************************************************************/
