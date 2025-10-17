/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 24, 2025.
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
#include <darwintest.h>

#undef _DARWIN_C_SOURCE
#undef _NONSTD_SOURCE
#define _POSIX_C_SOURCE 200112L

#include <sys/cdefs.h>
#include <sys/errno.h>
#include <sys/socket.h>

#include <netinet/in.h>

#include <stdio.h>
#include <string.h>

T_DECL(posix_so_linger, "POSIX SO_LINGER", T_META_TAG_VM_PREFERRED)
{
	T_LOG("POSIX SO_LINGER 0x%x", SO_LINGER);

	int s = -1;
	T_ASSERT_POSIX_SUCCESS(s = socket(AF_INET, SOCK_STREAM, 0),
	    "socket(AF_INET, SOCK_DGRAM)");

	struct linger set_l = { .l_onoff = 1, .l_linger = 5 };
	T_ASSERT_POSIX_SUCCESS(setsockopt(s, SOL_SOCKET, SO_LINGER,
	    &set_l, sizeof(struct linger)),
	    "setsockopt SO_LINGER");

	struct linger get_l = {};
	socklen_t len = sizeof(struct linger);
	T_ASSERT_POSIX_SUCCESS(getsockopt(s, SOL_SOCKET, SO_LINGER,
	    &get_l, &len),
	    "getsockopt SO_LINGER");

	T_EXPECT_EQ(set_l.l_onoff, get_l.l_onoff,
	    "POSIX SO_LINGER set l_onoff %d == get l_onoff %d",
	    set_l.l_onoff, get_l.l_onoff);
	T_EXPECT_EQ(set_l.l_linger, get_l.l_linger,
	    "POSIX SO_LINGER set l_linger %d == get l_linger %d",
	    set_l.l_linger, get_l.l_linger);
}

T_DECL(posix_so_linger_negative, "POSIX SO_LINGER negative", T_META_TAG_VM_PREFERRED)
{
	T_LOG("POSIX SO_LINGER 0x%x", SO_LINGER);

	int s = -1;
	T_ASSERT_POSIX_SUCCESS(s = socket(AF_INET, SOCK_STREAM, 0),
	    "socket(AF_INET, SOCK_DGRAM)");

	struct linger set_l = { .l_onoff = 0, .l_linger = -1 };
	T_ASSERT_POSIX_SUCCESS(setsockopt(s, SOL_SOCKET, SO_LINGER,
	    &set_l, sizeof(struct linger)),
	    "setsockopt SO_LINGER");

	struct linger get_l = {};
	socklen_t len = sizeof(struct linger);
	T_ASSERT_POSIX_SUCCESS(getsockopt(s, SOL_SOCKET, SO_LINGER,
	    &get_l, &len),
	    "getsockopt SO_LINGER");

	T_EXPECT_EQ(set_l.l_onoff, get_l.l_onoff,
	    "POSIX SO_LINGER set l_onoff %d == get l_onoff %d",
	    set_l.l_onoff, get_l.l_onoff);
	T_EXPECT_EQ(set_l.l_linger, get_l.l_linger,
	    "POSIX SO_LINGER set l_linger %d == get l_linger %d",
	    set_l.l_linger, get_l.l_linger);
}
