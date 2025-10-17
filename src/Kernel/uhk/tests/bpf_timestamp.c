/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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

T_DECL(bpf_no_timestamp, "test BIOCGNOTSTAMP and BIOCSNOTSTAMP")
{
	int fd = bpf_new();
	T_ASSERT_POSIX_SUCCESS(fd, "bpf open fd %d", fd);

	int get_no_timestamp = -1;

	T_ASSERT_POSIX_SUCCESS(ioctl(fd, BIOCGNOTSTAMP, &get_no_timestamp), "BIOCGNOTSTAMP");;
	T_LOG("BIOCGNOTSTAMP detault: %u", get_no_timestamp);

	int set_no_timestamp = 1;

	T_ASSERT_POSIX_SUCCESS(ioctl(fd, BIOCSNOTSTAMP, &set_no_timestamp), "BIOCSNOTSTAMP");;
	T_LOG("set_no_timestamp %u", set_no_timestamp);

	T_ASSERT_POSIX_SUCCESS(ioctl(fd, BIOCGNOTSTAMP, &get_no_timestamp), "BIOCGNOTSTAMP");;
	T_LOG("BIOCGNOTSTAMP detault: %u", get_no_timestamp);

	T_ASSERT_EQ(get_no_timestamp, set_no_timestamp, "get_no_timestamp == set_no_timestamp");

	set_no_timestamp = 0;

	T_ASSERT_POSIX_SUCCESS(ioctl(fd, BIOCSNOTSTAMP, &set_no_timestamp), "BIOCSNOTSTAMP");;
	T_LOG("set_no_timestamp %u", set_no_timestamp);

	T_ASSERT_POSIX_SUCCESS(ioctl(fd, BIOCGNOTSTAMP, &get_no_timestamp), "BIOCGNOTSTAMP");;
	T_LOG("get_no_timestamp %u", get_no_timestamp);

	T_ASSERT_EQ(get_no_timestamp, set_no_timestamp, "get_no_timestamp== set_no_timestamp");
}
