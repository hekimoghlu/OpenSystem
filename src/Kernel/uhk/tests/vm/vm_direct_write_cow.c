/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 27, 2023.
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
#include <darwintest_utils.h>

#include <errno.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.vm"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("VM"));

T_DECL(direct_write_cow,
    "test direct write() to file mapped with MAP_PRIVATE", T_META_TAG_VM_PREFERRED)
{
	int fd;
	int ret;
	char tmpf[PATH_MAX] = "";
	char *map_shared_addr, *map_private_addr;
	size_t file_size;
	ssize_t num_bytes;
	char *buf;

	T_SETUPBEGIN;

	strlcpy(tmpf, dt_tmpdir(), PATH_MAX);
	strlcat(tmpf, "/cow_direct_write.txt", PATH_MAX);
	T_LOG("file name: <%s>\n", tmpf);
	fd = open(tmpf, O_RDWR | O_CREAT | O_TRUNC, 0644);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(fd, "open()");

	file_size = PAGE_SIZE;
	ret = ftruncate(fd, (off_t) file_size);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(fd, "ftruncate()");

	ret = fcntl(fd, F_NOCACHE, true);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "fcntl(F_NOCACHE)");

	buf = valloc(file_size);
	T_QUIET; T_ASSERT_NOTNULL(buf, "valloc()");

	memset(buf, 'a', file_size);
	num_bytes = pwrite(fd, buf, file_size, 0);
	T_QUIET; T_EXPECT_POSIX_SUCCESS(num_bytes, "write()");
	T_QUIET; T_EXPECT_EQ(num_bytes, (ssize_t) file_size, "wrote <file_size> bytes");

	map_shared_addr = mmap(NULL, file_size, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(map_shared_addr, "mmap(MAP_SHARE)");
//	T_ASSERT_EQ(map_addr[0], 'a', 1); /* that would pollute the buffer cache... */

	map_private_addr = mmap(NULL, file_size, PROT_READ, MAP_FILE | MAP_PRIVATE, fd, 0);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(map_private_addr, "mmap(MAP_PRIVATE)");
//	T_ASSERT_EQ(map_addr[0], 'a', 1); /* that would pollute the buffer cache... */

	T_SETUPEND;

	memset(buf, 'b', file_size);
	num_bytes = pwrite(fd, buf, file_size, 0);
	T_QUIET; T_EXPECT_POSIX_SUCCESS(num_bytes, "write()");
	T_QUIET; T_EXPECT_EQ(num_bytes, (ssize_t) file_size, "wrote <file_size> bytes");

	T_ASSERT_EQ(map_shared_addr[0], 'b', "shared mapping was modified");
	T_ASSERT_EQ(map_private_addr[0], 'a', "private mapping was not modified");
}
