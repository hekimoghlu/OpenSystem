/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 1, 2025.
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
#include <unistd.h>

#include <darwintest.h>
#include <darwintest_utils.h>

T_GLOBAL_META(
	T_META_RUN_CONCURRENTLY(true),
	T_META_NAMESPACE("xnu.vfs"),
	T_META_OWNER("chrisjd")
	);

T_DECL(read_nullbuf, "read into a NULL buffer should fail")
{
	int fd;
	ssize_t result;

	fd = open("/etc/passwd", O_RDONLY);
	T_ASSERT_NE(fd, -1, "open /etc/passwd for reading");

	result = read(fd, NULL, 8);
	T_ASSERT_EQ(result, -1, "expected to fail");
	T_ASSERT_EQ(errno, EFAULT, "should have errno == EFAULT");
}
