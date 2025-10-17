/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <darwintest.h>


T_GLOBAL_META(
	T_META_NAMESPACE("xnu.tty"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("file descriptors"),
	T_META_OWNER("souvik_b"),
	T_META_RUN_CONCURRENTLY(true));

static void
tty_ioctl_tioccons(bool privileged)
{
	int primary;
	const char *name;
	int replica;
	int on = 1;
	int off = 0;

	// open primary tty
	T_ASSERT_POSIX_SUCCESS(primary = posix_openpt(O_RDWR | O_NOCTTY), "open primary");

	// allow opening a replica from the primary
	T_ASSERT_POSIX_SUCCESS(grantpt(primary), "grantpt");
	T_ASSERT_POSIX_SUCCESS(unlockpt(primary), "unlockpt");

	// get the name of the primary tty
	T_ASSERT_NOTNULL((name = ptsname(primary)), "ptsname");

	// open the replica
	T_ASSERT_POSIX_SUCCESS(replica = open(name, O_RDWR | O_NOCTTY), "open replica");

	// try calling the TIOCCONS ioctl
	if (privileged) {
		T_ASSERT_POSIX_SUCCESS(ioctl(primary, TIOCCONS, (char *)&on), "ioctl TIOCCONS on");
	} else {
		T_ASSERT_POSIX_ERROR(ioctl(primary, TIOCCONS, (char *)&on), -EPERM, "ioctl TIOCCONS on");
	}
	T_ASSERT_POSIX_SUCCESS(ioctl(primary, TIOCCONS, (char *)&off), "ioctl TIOCCONS off");

	// close primary and replica
	T_ASSERT_POSIX_SUCCESS(close(primary), "close primary");
	T_ASSERT_POSIX_SUCCESS(close(replica), "close replica");
}

T_DECL(tty_ioctl_tioccons_privileged,
    "call the TIOCCONS ioctl as root",
    T_META_ASROOT(true),
    T_META_TAG_VM_PREFERRED)
{
	tty_ioctl_tioccons(true);
}


T_DECL(tty_ioctl_tioccons_unprivileged,
    "call the TIOCCONS ioctl without root",
    T_META_ASROOT(false),
    T_META_TAG_VM_PREFERRED)
{
	tty_ioctl_tioccons(false);
}
