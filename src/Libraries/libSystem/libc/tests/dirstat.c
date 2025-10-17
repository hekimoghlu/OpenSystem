/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 24, 2022.
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

#include <stdlib.h>
#include <unistd.h>
#include <dirstat.h>
#include <darwintest.h>
#include <darwintest_utils.h>
#include <sys/stat.h>

#include <TargetConditionals.h>

#if !TARGET_OS_SIMULATOR
#define HAS_APFS
#endif

#ifdef HAS_APFS
#include <apfs/apfs_fsctl.h>
#endif

T_DECL(dirstat, "Runs dirstat(3)")
{
	bool fast_only = false;
	bool force_fallback = false;
	char *path = "/System/Library/Frameworks";

	// <rdar://problem/26400444> Libdarwintest passes argv without progname and flags starting in argv[0]
	optind = 0;

	int ch;
	while ((ch = getopt(argc, argv, "fup:")) != -1){
		switch (ch){
			case 'f':
				fast_only = true;
				break;
			case 'u':
				force_fallback = true;
				break;
			case 'p':
				path = optarg;
				break;
			case '?':
				T_ASSERT_FAIL("Usage: [-f] [-p <path>]");
		}
	}

	T_LOG("Path: %s", path);

	int flags = 0;
	if (fast_only) {
		T_LOG("Using DIRSTAT_FAST_ONLY");
		flags |= DIRSTAT_FAST_ONLY;
	}
	if (force_fallback) {
		T_LOG("Using DIRSTAT_FORCE_FALLBACK");
		flags |= DIRSTAT_FORCE_FALLBACK;
	}

	struct dirstat ds = {0};

	T_ASSERT_POSIX_SUCCESS(dirstat_np(path, flags, &ds, sizeof(ds)), NULL);

	T_LOG("Size: %zd bytes", ds.total_size);
	T_LOG("Descendants: %llu objects", ds.descendants);
}

T_DECL(dirstat_fallback, "Tests dirstat(3) fallback")
{
	char *path = "/System/Library/Frameworks";

	struct dirstat ds = {0};

	off_t native_size = 0;
	off_t fallback_size = 0;

	T_LOG("dirstat_np(\"%s\", 0, ...)", path);
	T_EXPECT_POSIX_SUCCESS(dirstat_np(path, 0, &ds, sizeof(ds)), NULL);
	T_LOG("Size: %zd bytes", ds.total_size);
	T_LOG("Descendants: %llu objects", ds.descendants);
	native_size = ds.total_size;

	T_LOG("dirstat_np(\"%s\", DIRSTAT_FORCE_FALLBACK, ...)", path);
	T_EXPECT_POSIX_SUCCESS(dirstat_np(path, DIRSTAT_FORCE_FALLBACK, &ds, sizeof(ds)), NULL);
	T_LOG("Size: %zd bytes", ds.total_size);
	T_LOG("Descendants: %llu objects", ds.descendants);
	fallback_size = ds.total_size;

	T_EXPECT_EQ(native_size, fallback_size, "Native and fallback sizes match");
}

T_DECL(dirstat_flags, "Tests dirstat(3)'s behavior flags")
{
	int err;
	struct dirstat ds = {0};

	char *not_fast = "/System/Library/Frameworks";

	T_LOG("dirstat_np(\"%s\", DIRSTAT_FAST_ONLY, ...)", not_fast);
	T_EXPECT_EQ(dirstat_np(not_fast, DIRSTAT_FAST_ONLY, &ds, sizeof(ds)), -1, "Fast-only fails on non-fast-enabled directory");

#ifdef HAS_APFS
	char *fast_tmpdir;
	uint64_t flags = 0;

	T_SETUPBEGIN;
	T_ASSERT_NE(asprintf(&fast_tmpdir, "%s/%s", dt_tmpdir(), "fast-XXXXXX"), -1, "Generate fast dir stats directory name");
	T_ASSERT_EQ((void *)mkdtemp(fast_tmpdir), (void *)fast_tmpdir, "Make fast dir stats directory");
	err = fsctl(fast_tmpdir, APFSIOC_MAINTAIN_DIR_STATS, &flags, 0);
	if (err != 0) {
		rmdir(fast_tmpdir);
		free(fast_tmpdir);
		T_SKIP("Couldn't enable fast dir stats for directory (not on APFS?)");
	}
	T_SETUPEND;

	T_LOG("dirstat_np(\"%s\", DIRSTAT_FAST_ONLY, ...)", fast_tmpdir);
	T_EXPECTFAIL;
	T_EXPECT_POSIX_SUCCESS(dirstat_np(fast_tmpdir, DIRSTAT_FAST_ONLY, &ds, sizeof(ds)), "Fast-only works on fast-enabled directory");

	T_ASSERT_POSIX_SUCCESS(rmdir(fast_tmpdir), NULL);
	free(fast_tmpdir);
#endif
}
