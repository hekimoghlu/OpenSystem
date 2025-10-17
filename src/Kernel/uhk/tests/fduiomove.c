/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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
#include <crt_externs.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <fcntl.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.fd"),
	T_META_RUN_CONCURRENTLY(true));

T_DECL(fd_invalid_pread, "Test for 66711697: make sure we get EFAULT", T_META_TAG_VM_PREFERRED)
{
	int fd;
	ssize_t rc;

	fd = open(*_NSGetProgname(), O_RDONLY);
	T_ASSERT_POSIX_SUCCESS(fd, "open(self)");

	rc = pread(fd, (void *)~0, 64 << 10, 0);
	T_ASSERT_POSIX_FAILURE(rc, EFAULT, "pread should fail with EFAULT");

	close(fd);
}
