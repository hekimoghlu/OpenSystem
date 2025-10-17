/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 27, 2024.
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
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <mach/clock_types.h>
#include <sys/mman.h>
#include <sys/timex.h>
#include <spawn.h>
#include <darwintest.h>
#include <darwintest_utils.h>

/*
 * This test expects the entitlement or root privileges for a process to
 * set the time using settimeofday syscall.
 */

#define DAY 86400 //1 day in sec

T_DECL(settime_32089962_not_entitled_root,
    "Verify that root privileges can allow to change the time",
    T_META_ASROOT(true), T_META_CHECK_LEAKS(false))
{
	struct timeval settimeofdaytime;
	struct timeval adj_time;
	struct timex ntptime;

	/* test settimeofday */
	T_QUIET; T_ASSERT_POSIX_ZERO(gettimeofday(&settimeofdaytime, NULL), NULL);
	T_ASSERT_POSIX_ZERO(settimeofday(&settimeofdaytime, NULL), NULL);

	/* test adjtime */
	adj_time.tv_sec = 1;
	adj_time.tv_usec = 0;
	T_ASSERT_POSIX_ZERO(adjtime(&adj_time, NULL), NULL);

	/* test ntp_adjtime */
	memset(&ntptime, 0, sizeof(ntptime));
	ntptime.modes |= MOD_STATUS;
	ntptime.status = TIME_OK;

	T_ASSERT_EQ(ntp_adjtime(&ntptime), TIME_OK, NULL);
}

T_DECL(settime_32089962_not_entitled_not_root,
    "Verify that the \"com.apple.settime\" entitlement can allow to change the time",
    T_META_ASROOT(false), T_META_CHECK_LEAKS(false))
{
	struct timeval settimeofdaytime;
	struct timeval adj_time;
	struct timex ntptime;
	int res;

	if (geteuid() == 0) {
		T_SKIP("settimeofday_29193041 test requires no root privileges to run.");
	}

	T_QUIET; T_ASSERT_POSIX_ZERO(gettimeofday(&settimeofdaytime, NULL), NULL);

	/* test settimeofday */
#if (TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR)
	T_ASSERT_POSIX_ZERO(settimeofday(&settimeofdaytime, NULL), NULL);
#else
	res = settimeofday(&settimeofdaytime, NULL);
	T_ASSERT_EQ(res, -1, NULL);
#endif

	/* test adjtime */
	adj_time.tv_sec = 1;
	adj_time.tv_usec = 0;
	res = adjtime(&adj_time, NULL);
	T_ASSERT_EQ(res, -1, NULL);

	/* test ntp_adjtime */
	memset(&ntptime, 0, sizeof(ntptime));
	ntptime.modes |= MOD_STATUS;
	ntptime.status = TIME_OK;
	res = ntp_adjtime(&ntptime);
	T_ASSERT_EQ(res, -1, NULL);
}

T_DECL(settimeofday_29193041_not_entitled_root,
    "Verify that root privileges can allow to change the time",
    T_META_ASROOT(true), T_META_CHECK_LEAKS(false))
{
	struct timeval time;
	long new_time;

	T_QUIET; T_ASSERT_POSIX_ZERO(gettimeofday(&time, NULL), NULL);

	/* increment the time of one day */
	new_time = time.tv_sec + DAY;

	time.tv_sec = new_time;
	time.tv_usec = 0;

	T_ASSERT_POSIX_ZERO(settimeofday(&time, NULL), NULL);

	T_QUIET; T_ASSERT_POSIX_ZERO(gettimeofday(&time, NULL), NULL);

	/* expext to be past new_time */
	T_EXPECT_GE_LONG(time.tv_sec, new_time, "Time changed with root and without entitlement");

	time.tv_sec -= DAY;
	T_QUIET; T_ASSERT_POSIX_ZERO(settimeofday(&time, NULL), NULL);
}

T_DECL(settimeofday_29193041_not_entitled_not_root,
    "Verify that the \"com.apple.settime\" entitlement can allow to change the time",
    T_META_ASROOT(false), T_META_CHECK_LEAKS(false))
{
	struct timeval time;
	long new_time;

	if (geteuid() == 0) {
		T_SKIP("settimeofday_29193041 test requires no root privileges to run.");
	}

	T_QUIET; T_ASSERT_POSIX_ZERO(gettimeofday(&time, NULL), NULL);

	/* increment the time of one day */
	new_time = time.tv_sec + DAY;

	time.tv_sec = new_time;
	time.tv_usec = 0;

#if (TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR)
	T_ASSERT_POSIX_ZERO(settimeofday(&time, NULL), NULL);
#else
	int res = settimeofday(&time, NULL);
	T_ASSERT_EQ(res, -1, NULL);
#endif

	T_QUIET; T_ASSERT_POSIX_ZERO(gettimeofday(&time, NULL), NULL);

#if (TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR)
	/* expext to be past new_time */
	T_EXPECT_GE_LONG(time.tv_sec, new_time, "Time successfully changed without root and without entitlement");
	time.tv_sec -= DAY;
	T_QUIET; T_ASSERT_POSIX_ZERO(settimeofday(&time, NULL), NULL);
#else
	T_EXPECT_LT_LONG(time.tv_sec, new_time, "Not permitted to change time without root and without entitlement");
#endif
}
