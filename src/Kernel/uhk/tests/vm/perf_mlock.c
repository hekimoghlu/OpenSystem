/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 13, 2025.
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
#include <darwintest_perf.h>
#include <darwintest_utils.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <sys/syslimits.h>

/*
 * Wiring performance micro-benchmark.
 */

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.vm.perf"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("VM"),
	T_META_OWNER("jarrad"),
	T_META_CHECK_LEAKS(false),
	T_META_TAG_PERF,
	T_META_REQUIRE_NOT_VIRTUALIZED);

#define MiB(b) ((uint64_t)b << 20)

extern int vfs_purge(void);

T_DECL(mlock_external,
    "File-backed wire microbenchmark",
    T_META_ENABLED(false /* rdar://133954365 */))
{
	void *buf = NULL;
	int fd, ret;
	char tmpf[PATH_MAX] = "";
	uint8_t pattern[4] = {0xFEu, 0xDCu, 0xBAu, 0x98u};
	// This should be kept larger than MAX_UPL_TRANFER_BYTES to ensure clustering
	// greater than the default fault clustering is tested
	const size_t vmsize = MiB(512);

	T_SETUPBEGIN;

	buf = malloc(vmsize);
	T_QUIET; T_ASSERT_NOTNULL(buf, "malloc()");

	// Create a tmp file and populate it with data
	strlcpy(tmpf, dt_tmpdir(), PATH_MAX);
	strlcat(tmpf, "/mlock_external.txt", PATH_MAX);
	T_QUIET; T_ASSERT_LT(strlen(tmpf), (unsigned long)PATH_MAX,
	    "path exceeds PATH_MAX");

	fd = open(tmpf, (O_RDWR | O_CREAT));
	T_QUIET; T_ASSERT_POSIX_SUCCESS(fd, "open()");

	memset_pattern4((char *)buf, pattern, vmsize);
	write(fd, buf, vmsize);
	fsync(fd);
	close(fd);
	free(buf);
	buf = NULL;

	// Purge to flush the test file from the filecache
	vfs_purge();

	fd = open(tmpf, O_RDWR);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(fd, "open()");
	buf = mmap(NULL, vmsize, PROT_READ,
	    (MAP_FILE | MAP_SHARED), fd, 0);
	T_QUIET; T_ASSERT_NOTNULL(buf, "mmap()");

	T_SETUPEND;

	dt_stat_time_t mlock_time = dt_stat_time_create("mlock_duration");
	dt_stat_set_variable(mlock_time, "buffer_size_mb",
	    (unsigned int)(vmsize >> 20));

	T_LOG("Collecting measurements...");
	while (!dt_stat_stable(mlock_time)) {
		T_STAT_MEASURE(mlock_time) {
			ret = mlock(buf, vmsize);
		}
		T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "mlock()");

		ret = munlock(buf, vmsize);
		T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "munlock()");

		vfs_purge();
	}
	dt_stat_finalize(mlock_time);

	ret = munmap(buf, vmsize);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "munmap()");
	ret = close(fd);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(ret, "close()");
}
