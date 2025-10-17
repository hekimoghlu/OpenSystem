/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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
#include <fcntl.h>
#include <sys/fcntl.h>
#include <darwintest.h>
#include <darwintest_utils.h>


T_GLOBAL_META(
	T_META_NAMESPACE("xnu.ipc"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("IPC"),
	T_META_OWNER("jonathan_w_adams"),
	T_META_RUN_CONCURRENTLY(TRUE));

/*
 * See rdar://77264182: xnu's lockf implementation had trouble
 * with l_len = 0 (e.g. go to EOF) being treated differently
 * than (l_start + l_len - 1) == OFF_MAX, even though they are
 * effectively the same thing.  ~25 loops of this test was enough
 * to get an Intel mac into an infinite loop in the kernel.
 */
T_DECL(lockf_EOF_77264182,
    "try to stress out lockf requests around OFF_MAX/EOF",
    T_META_CHECK_LEAKS(false), T_META_TAG_VM_PREFERRED)
{
	const char *dir = dt_tmpdir();
	int fd;
	T_ASSERT_POSIX_SUCCESS(chdir(dir), "chdir(%s)", dir);

	T_ASSERT_POSIX_SUCCESS((fd = open("lockf_EOF_test", O_CREAT | O_RDWR, 0666)), "open(lockf_EOF_test)");

	/*
	 * At each loop, we do:
	 *	write lock [OFF_MAX - loop, EOF)
	 *	unlock     [OFF_MAX - loop, OFF_MAX)
	 *	write lock [OFF_MAX - loop - 1, OFF_MAX)
	 */
	int loops;
	for (loops = 0; loops < 100; loops++) {
		struct flock fl = {
			.l_start = OFF_MAX - loops,
			.l_len = 0,
			.l_pid = getpid(),
			.l_type = F_WRLCK,
			.l_whence = SEEK_SET
		};
		T_ASSERT_POSIX_SUCCESS(fcntl(fd, F_SETLK, &fl), "wrlock");
		fl.l_len = OFF_MAX - fl.l_start + 1;
		fl.l_type = F_UNLCK;
		T_ASSERT_POSIX_SUCCESS(fcntl(fd, F_SETLK, &fl), "unlock");
		fl.l_start--;
		fl.l_len++;
		fl.l_type = F_WRLCK;
		T_ASSERT_POSIX_SUCCESS(fcntl(fd, F_SETLK, &fl), "wrlock 2");
	}
	T_PASS("did %d loops", loops);
}
