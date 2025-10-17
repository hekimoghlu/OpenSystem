/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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
 * Regress test for misc poll/ppoll timeout helpers.
 *
 * Placed in the public domain.
 */

#include "includes.h"

#include <sys/types.h>
#include <stdio.h>
#ifdef HAVE_STDINT_H
# include <stdint.h>
#endif
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_POLL_H
# include <poll.h>
#endif
#include <time.h>

#include "../test_helper/test_helper.h"

#include "log.h"
#include "misc.h"

void test_ptimeout(void);

void
test_ptimeout(void)
{
	struct timespec pt, *ts;

	TEST_START("ptimeout_init");
	ptimeout_init(&pt);
	ASSERT_PTR_EQ(ptimeout_get_tsp(&pt), NULL);
	ASSERT_INT_EQ(ptimeout_get_ms(&pt), -1);
	TEST_DONE();

	TEST_START("ptimeout_deadline_sec");
	ptimeout_deadline_sec(&pt, 100);
	ptimeout_deadline_sec(&pt, 200);
	ASSERT_INT_EQ(ptimeout_get_ms(&pt), 100 * 1000);
	ts = ptimeout_get_tsp(&pt);
	ASSERT_PTR_NE(ts, NULL);
	ASSERT_LONG_EQ(ts->tv_nsec, 0);
	ASSERT_LONG_EQ(ts->tv_sec, 100);
	TEST_DONE();

	TEST_START("ptimeout_deadline_ms");
	ptimeout_deadline_ms(&pt, 50123);
	ptimeout_deadline_ms(&pt, 50500);
	ASSERT_INT_EQ(ptimeout_get_ms(&pt), 50123);
	ts = ptimeout_get_tsp(&pt);
	ASSERT_PTR_NE(ts, NULL);
	ASSERT_LONG_EQ(ts->tv_nsec, 123 * 1000000);
	ASSERT_LONG_EQ(ts->tv_sec, 50);
	TEST_DONE();

	TEST_START("ptimeout zero");
	ptimeout_init(&pt);
	ptimeout_deadline_ms(&pt, 0);
	ASSERT_INT_EQ(ptimeout_get_ms(&pt), 0);
	ts = ptimeout_get_tsp(&pt);
	ASSERT_PTR_NE(ts, NULL);
	ASSERT_LONG_EQ(ts->tv_nsec, 0);
	ASSERT_LONG_EQ(ts->tv_sec, 0);
	TEST_DONE();

	TEST_START("ptimeout_deadline_monotime");
	ptimeout_init(&pt);
	ptimeout_deadline_monotime(&pt, monotime() + 100);
	ASSERT_INT_GT(ptimeout_get_ms(&pt), 50000);
	ASSERT_INT_LT(ptimeout_get_ms(&pt), 200000);
	ts = ptimeout_get_tsp(&pt);
	ASSERT_PTR_NE(ts, NULL);
	ASSERT_LONG_GT(ts->tv_sec, 50);
	ASSERT_LONG_LT(ts->tv_sec, 200);
	TEST_DONE();

	TEST_START("ptimeout_deadline_monotime past");
	ptimeout_init(&pt);
	ptimeout_deadline_monotime(&pt, monotime() + 100);
	ptimeout_deadline_monotime(&pt, monotime() - 100);
	ASSERT_INT_EQ(ptimeout_get_ms(&pt), 0);
	ts = ptimeout_get_tsp(&pt);
	ASSERT_PTR_NE(ts, NULL);
	ASSERT_LONG_EQ(ts->tv_nsec, 0);
	ASSERT_LONG_EQ(ts->tv_sec, 0);
	TEST_DONE();
}
