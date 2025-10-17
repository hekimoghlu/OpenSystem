/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 26, 2023.
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
#include <dispatch/dispatch.h>
#include <execinfo.h>
#include <pthread.h>
#include <ptrauth.h>
#include <mach/mach.h>
#include <stdalign.h>
#include <sys/mman.h>
#include <sys/sysctl.h>


static const off_t gb = 1024 * 1024 * 1024;

static void
get_release_type(char *release_type, size_t release_type_len)
{
	int ret;
	T_ASSERT_POSIX_SUCCESS(ret = sysctlbyname("kern.osreleasetype", release_type, &release_type_len, NULL, 0), "sysctlbyname kern.osreleasetype");
}

static int64_t
get_hw_memsize(int64_t memsize)
{
	int ret;
	size_t size = sizeof(memsize);
	T_ASSERT_POSIX_SUCCESS(ret = sysctlbyname("hw.memsize", &memsize, &size, NULL, 0), "sysctlbyname hw.memsize");

	return memsize;
}

T_DECL(big_map_test,
    "Test that loads large blobs into memory up to 60 percent of gigs available.",
    T_META_ASROOT(true),
    T_META_CHECK_LEAKS(false))
{
	int fd;
	int64_t memsize = 0;
	size_t release_type_len = 256;
	char release_type[release_type_len];
	const char required_release_type[] = "Darwin Cloud";

	get_release_type(release_type, release_type_len);
	if (strstr(release_type, required_release_type) == NULL) {
		T_SKIP("Attempted to run on non psOS release type, skipping...");
	}

	memsize = get_hw_memsize(memsize);
	float max_memory_gib = ((float)memsize / (float)gb) * .6;

	if (max_memory_gib <= 11) {
		T_SKIP("Not enough memory on device atleast (11GBs required), skipping...");
	}

	char file_path[] = "/tmp/bigfile.XXXXXX";
	T_QUIET; T_ASSERT_POSIX_SUCCESS(fd = mkstemp(file_path), NULL);
	for (int gigs = 1; gigs <= max_memory_gib; gigs++) {
		size_t bytes = gigs * gb;

		T_LOG("trying %zu bytes (%d GB)\n", bytes, gigs);

		T_QUIET; T_ASSERT_POSIX_SUCCESS(ftruncate(fd, bytes), "ftruncate");

		void *p = mmap(NULL, bytes, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0);

		T_QUIET; T_ASSERT_NE(p, MAP_FAILED, "map");

		T_QUIET; T_ASSERT_POSIX_SUCCESS(munmap(p, bytes), "munmap");
	}
}
