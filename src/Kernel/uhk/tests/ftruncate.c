/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <sys/resource.h>
#include <unistd.h>

T_GLOBAL_META(T_META_NAMESPACE("xnu.vfs"));

#define FSIZE_CUR (10*1024)
#define TMP_FILE_PATH "/tmp/ftruncate_test"

static int sigcount = 0;

static void
xfsz_signal_handler(__unused int signo)
{
	sigcount++;
}

static void
fsize_test(bool use_fd)
{
	struct rlimit rlim;
	int fd, ret;

	T_SETUPBEGIN;

	signal(SIGXFSZ, xfsz_signal_handler);

	rlim.rlim_cur = FSIZE_CUR;
	rlim.rlim_max = RLIM_INFINITY;
	ret = setrlimit(RLIMIT_FSIZE, &rlim);
	T_ASSERT_POSIX_SUCCESS(ret, "set soft RLIMIT_FSIZE to %d", FSIZE_CUR);

	fd = open(TMP_FILE_PATH, O_RDWR | O_CREAT, 0777);
	T_ASSERT_POSIX_SUCCESS(ret, "create temp file: %s", TMP_FILE_PATH);

	T_SETUPEND;

	if (use_fd) {
		ret = ftruncate(fd, FSIZE_CUR);
		T_EXPECT_POSIX_SUCCESS(ret, "ftruncate() with length RLIMIT_FSIZE");
	} else {
		ret = truncate(TMP_FILE_PATH, FSIZE_CUR);
		T_EXPECT_POSIX_SUCCESS(ret, "truncate() with length RLIMIT_FSIZE");
	}
	T_EXPECT_EQ(sigcount, 0, "no signal received");

	if (use_fd) {
		ret = ftruncate(fd, FSIZE_CUR + 1);
		T_EXPECT_POSIX_FAILURE(ret, EFBIG, "ftruncate() with length RLIMIT_FSIZE + 1");
	} else {
		ret = truncate(TMP_FILE_PATH, FSIZE_CUR + 1);
		T_EXPECT_POSIX_FAILURE(ret, EFBIG, "truncate() with length RLIMIT_FSIZE + 1");
	}
	T_EXPECT_EQ(sigcount, 1, "SIGXFSZ signal received");

	ret = close(fd);
	T_ASSERT_POSIX_SUCCESS(ret, "close temp file");

	ret = unlink(TMP_FILE_PATH);
	T_ASSERT_POSIX_SUCCESS(ret, "unlink temp file");
}

T_DECL(ftruncate_fsize,
    "ftruncate() should fail with EFBIG and send SIGXFSZ signal when length > RLIMIT_FSIZE", T_META_TAG_VM_PREFERRED)
{
	fsize_test(true);
}

T_DECL(truncate_fsize,
    "truncate() should fail with EFBIG and send SIGXFSZ signal when length > RLIMIT_FSIZE", T_META_TAG_VM_PREFERRED)
{
	fsize_test(false);
}
