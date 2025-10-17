/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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
#include <sys/socket.h>
#include <sys/sysctl.h>
#include <sys/time.h>

#include <darwintest.h>

T_DECL(so_linger_negative, "SO_LINGER negative")
{
	int s = -1;
	T_ASSERT_POSIX_SUCCESS(s = socket(AF_LOCAL, SOCK_DGRAM, 0), "socket(AF_LOCAL, SOCK_DGRAM)");

	struct linger set_l = {};
	set_l.l_onoff = 1;
	set_l.l_linger = -1;
	T_LOG("l_linger %d", set_l.l_linger);
	T_ASSERT_POSIX_SUCCESS(setsockopt(s, SOL_SOCKET, SO_LINGER,
	    &set_l, sizeof(struct linger)),
	    "setsockopt SO_LINGER");

	struct linger get_l = {};
	socklen_t len = sizeof(struct linger);
	T_ASSERT_POSIX_SUCCESS(getsockopt(s, SOL_SOCKET, SO_LINGER,
	    &get_l, &len),
	    "getsockopt SO_LINGER");

	T_EXPECT_EQ(set_l.l_linger, get_l.l_linger,
	    "SO_LINGER negative l_linger %d == %d", set_l.l_linger, get_l.l_linger);
}

T_DECL(so_linger_overflow, "SO_LINGER overflow")
{
	int s = -1;
	T_ASSERT_POSIX_SUCCESS(s = socket(AF_LOCAL, SOCK_DGRAM, 0), "socket(AF_LOCAL, SOCK_DGRAM)");

	struct linger set_l = {};
	set_l.l_onoff = 1;
	set_l.l_linger = SHRT_MAX + 1;
	T_LOG("l_linger %d", set_l.l_linger);
	T_ASSERT_POSIX_SUCCESS(setsockopt(s, SOL_SOCKET, SO_LINGER,
	    &set_l, sizeof(struct linger)),
	    "setsockopt SO_LINGER");

	struct linger get_l = {};
	socklen_t len = sizeof(struct linger);
	T_ASSERT_POSIX_SUCCESS(getsockopt(s, SOL_SOCKET, SO_LINGER,
	    &get_l, &len),
	    "getsockopt SO_LINGER");

	/*
	 * Test passes based on the knowledge that l_linger is stored
	 * as a short signed integer
	 */
	T_EXPECT_EQ((short)set_l.l_linger, (short)get_l.l_linger,
	    "SO_LINGER overflow l_linger (short) %d == (short) %d",
	    set_l.l_linger, get_l.l_linger);
}

T_DECL(so_linger_500, "SO_LINGER 500")
{
	int s = -1;
	T_ASSERT_POSIX_SUCCESS(s = socket(AF_LOCAL, SOCK_DGRAM, 0), "socket(AF_LOCAL, SOCK_DGRAM)");

	struct clockinfo clkinfo;
	size_t oldlen = sizeof(struct clockinfo);
	T_ASSERT_POSIX_SUCCESS(sysctlbyname("kern.clockrate", &clkinfo, &oldlen, NULL, 0),
	    "sysctlbyname(kern.clockrate)");

	struct linger set_l = {};
	set_l.l_onoff = 1;
	set_l.l_linger = 500;

	T_ASSERT_POSIX_SUCCESS(setsockopt(s, SOL_SOCKET, SO_LINGER,
	    &set_l, sizeof(struct linger)),
	    "setsockopt SO_LINGER");

	struct linger get_l = {};
	socklen_t len = sizeof(struct linger);
	T_ASSERT_POSIX_SUCCESS(getsockopt(s, SOL_SOCKET, SO_LINGER,
	    &get_l, &len),
	    "getsockopt SO_LINGER");

	T_EXPECT_EQ(set_l.l_linger, get_l.l_linger,
	    "SO_LINGER 500 l_linger %d == %d", set_l.l_linger, get_l.l_linger);
}

T_DECL(so_linger_sec_negative, "SO_LINGER_SEC negative")
{
	int s = -1;
	T_ASSERT_POSIX_SUCCESS(s = socket(AF_LOCAL, SOCK_DGRAM, 0), "socket(AF_LOCAL, SOCK_DGRAM)");

	struct clockinfo clkinfo;
	size_t oldlen = sizeof(struct clockinfo);
	T_ASSERT_POSIX_SUCCESS(sysctlbyname("kern.clockrate", &clkinfo, &oldlen, NULL, 0),
	    "sysctlbyname(kern.clockrate)");

	struct linger set_l = {};
	set_l.l_onoff = 1;
	set_l.l_linger = -1;
	T_LOG("l_linger %d * clkinfo.hz %d = %d", set_l.l_linger, clkinfo.hz, (set_l.l_linger * clkinfo.hz));

	T_ASSERT_POSIX_SUCCESS(setsockopt(s, SOL_SOCKET, SO_LINGER_SEC,
	    &set_l, sizeof(struct linger)),
	    "setsockopt SO_LINGER_SEC");

	struct linger get_l = {};
	socklen_t len = sizeof(struct linger);
	T_ASSERT_POSIX_SUCCESS(getsockopt(s, SOL_SOCKET, SO_LINGER_SEC,
	    &get_l, &len),
	    "getsockopt SO_LINGER_SEC");

	T_EXPECT_EQ(set_l.l_linger, get_l.l_linger,
	    "SO_LINGER_SEC negative l_linger %d == %d", set_l.l_linger, get_l.l_linger);
}

T_DECL(so_linger_sec_5_seconds, "SO_LINGER_SEC 5 seconds")
{
	int s = -1;
	T_ASSERT_POSIX_SUCCESS(s = socket(AF_LOCAL, SOCK_DGRAM, 0), "socket(AF_LOCAL, SOCK_DGRAM)");

	struct clockinfo clkinfo;
	size_t oldlen = sizeof(struct clockinfo);
	T_ASSERT_POSIX_SUCCESS(sysctlbyname("kern.clockrate", &clkinfo, &oldlen, NULL, 0),
	    "sysctlbyname(kern.clockrate)");

	struct linger set_l = {};
	set_l.l_onoff = 1;
	set_l.l_linger = 5;
	T_LOG("l_linger %d * clkinfo.hz %d = %d", set_l.l_linger, clkinfo.hz, (set_l.l_linger * clkinfo.hz));

	T_ASSERT_POSIX_SUCCESS(setsockopt(s, SOL_SOCKET, SO_LINGER_SEC,
	    &set_l, sizeof(struct linger)),
	    "setsockopt SO_LINGER_SEC");

	struct linger get_l = {};
	socklen_t len = sizeof(struct linger);
	T_ASSERT_POSIX_SUCCESS(getsockopt(s, SOL_SOCKET, SO_LINGER_SEC,
	    &get_l, &len),
	    "getsockopt SO_LINGER_SEC");

	T_EXPECT_EQ(set_l.l_linger, get_l.l_linger,
	    "SO_LINGER_SEC 5 seconds l_linger %d == %d", set_l.l_linger, get_l.l_linger);
}
