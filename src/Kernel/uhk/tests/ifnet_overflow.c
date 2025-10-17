/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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
#include <sys/socket.h>
#include <net/if.h>
#include <sys/ioctl.h>

static void __unused
create_interfaces(const char *prefix, int num)
{
	static int fd = -1;

	if (fd == -1) {
		fd = socket(PF_INET, SOCK_STREAM, 0);
		T_QUIET; T_ASSERT_POSIX_SUCCESS(fd, "socket");
	}
	for (int i = 0; i < num; i++) {
		struct ifreq ifr = {};

		sprintf(ifr.ifr_name, "%s%d", prefix, i);
		int ret = ioctl(fd, SIOCIFCREATE, &ifr);
		if (errno == EEXIST) {
			continue;
		}
		T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "SIOCIFCREATE");
		memset(&ifr, 0, sizeof(ifr));
		sprintf(ifr.ifr_name, "%s%d", prefix, i);
		ret = ioctl(fd, SIOCIFDESTROY, &ifr);
		T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "SIOCIFDESTROY");
		if (i % 100 == 0) {
			printf("created %s%d\n", prefix, i);
		}
	}
}


T_DECL(test_ifnet_overflow,
    "Verify that we don't crash when we create many interfaces",
    T_META_CHECK_LEAKS(false), T_META_TAG_VM_PREFERRED)
{
#if 1
	T_SKIP("Not stable yet");
#else
	create_interfaces("vlan", 32768);
	create_interfaces("feth", 32768);
#endif
}
