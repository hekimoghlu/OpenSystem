/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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

#include <sys/ioctl.h>

#include <net/bpf.h>

#include "bpflib.h"

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.net"),
	T_META_ASROOT(true),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("networking"),
	T_META_CHECK_LEAKS(false));

#ifdef BIOCGDIRECTION

static const char *
str_bpf_direction(u_int direction)
{
	switch (direction) {
	case BPF_D_NONE:
		return "BPF_D_NONE";
	case BPF_D_IN:
		return "BPF_D_IN";
	case BPF_D_OUT:
		return "BPF_D_OUT";
	case BPF_D_INOUT:
		return "BPF_D_INOUT";
	default:
		break;
	}
	return "<invalid>";
}

static void
test_set_direction(int fd, u_int direction)
{
	T_ASSERT_POSIX_SUCCESS(bpf_set_direction(fd, direction),
	    "bpf_set_direction(%d): %u (%s)", fd, direction, str_bpf_direction(direction));

	u_int get_direction = (u_int)(-2);
	T_ASSERT_POSIX_SUCCESS(bpf_get_direction(fd, &get_direction),
	    "bpf_get_direction(%d): %u (%s)", fd, get_direction, str_bpf_direction(get_direction));
	T_ASSERT_EQ(get_direction, direction, "get_direction %d == direction %d", get_direction, direction);
}

T_DECL(bpf_direction, "test BPF set and grep direction", T_META_TAG_VM_PREFERRED)
{
	int fd = bpf_new();
	T_ASSERT_POSIX_SUCCESS(fd, "bpf open fd %d", fd);

	u_int direction = (u_int)(-2); /* an invalid value */
	T_ASSERT_POSIX_SUCCESS(bpf_get_direction(fd, &direction),
	    "bpf_get_direction(%d): %u (%s)", fd, direction, str_bpf_direction(direction));
	T_ASSERT_EQ(direction, BPF_D_INOUT, "initial direction not BPF_D_INOUT");

	test_set_direction(fd, BPF_D_INOUT);
	test_set_direction(fd, BPF_D_IN);
	test_set_direction(fd, BPF_D_OUT);
	test_set_direction(fd, BPF_D_NONE);

	direction = 10;
	T_EXPECT_POSIX_FAILURE(bpf_set_direction(fd, direction), EINVAL,
	    "bpf_set_direction(%d): %u (%s)", fd, direction, str_bpf_direction(direction));
}

#else /* BIOCSETDIRECTION */

T_DECL(bpf_direction, "test BPF set and grep direction", T_META_TAG_VM_PREFERRED)
{
	T_SKIP("BIOCSETDIRECTION is not defined");
}

#endif /* BIOCSETDIRECTION */

T_DECL(bpf_seesent, "test BIOCGSEESENT and BIOCSSEESENT", T_META_TAG_VM_PREFERRED)
{
	int fd = bpf_new();
	T_ASSERT_POSIX_SUCCESS(fd, "bpf open fd %d", fd);

	u_int get_see_sent = (u_int) - 1;

	T_ASSERT_POSIX_SUCCESS(ioctl(fd, BIOCGSEESENT, &get_see_sent), "BIOCGSEESENT");;
	T_LOG("get_see_sent %u", get_see_sent);

	u_int set_see_sent = get_see_sent == 0 ? 1 : 0;

	T_ASSERT_POSIX_SUCCESS(ioctl(fd, BIOCSSEESENT, &set_see_sent), "BIOCSSEESENT");;
	T_LOG("set_see_sent %u", set_see_sent);

	T_ASSERT_POSIX_SUCCESS(ioctl(fd, BIOCGSEESENT, &get_see_sent), "BIOCGSEESENT");;
	T_LOG("get_see_sent %u", set_see_sent);

	T_ASSERT_EQ(get_see_sent, set_see_sent, "get_see_sent == set_see_sent");

	set_see_sent = get_see_sent == 0 ? 1 : 0;

	T_ASSERT_POSIX_SUCCESS(ioctl(fd, BIOCSSEESENT, &set_see_sent), "BIOCSSEESENT");;
	T_LOG("set_see_sent %u", set_see_sent);

	T_ASSERT_POSIX_SUCCESS(ioctl(fd, BIOCGSEESENT, &get_see_sent), "BIOCGSEESENT");;
	T_LOG("get_see_sent %u", get_see_sent);

	T_ASSERT_EQ(get_see_sent, set_see_sent, "get_see_sent == set_see_sent");
}
