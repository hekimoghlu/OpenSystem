/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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
#include <unistd.h>
#include <sys/stat.h>


#include "hfs-tests.h"
#include "test-utils.h"
#include "disk-image.h"

#define SYMPLINK_TEST_DIR "symlink.testdir"
#define SYMLINK_EMPTYSTR "symlink.emptystr"
TEST(symlinks)

int run_symlinks(__unused test_ctx_t *ctx)
{
	disk_image_t *di;
	struct stat statb;
	char *parent_dir, *slink;
	char buf;

	di = disk_image_get();

	//
	// Create a parent directory to host our test.
	//
	asprintf(&parent_dir, "%s/"SYMPLINK_TEST_DIR, di->mount_point);
	assert(!mkdir(parent_dir, 0777) || errno == EEXIST);

	//
	// Now check to make sure we support creating a symlink with an empty
	// target required for UNIX Conformance.
	//
	asprintf(&slink, "%s/"SYMLINK_EMPTYSTR, parent_dir);
	assert_no_err(symlink("", slink));

	//
	// Test that symlink has "l" as the S_ISLNK flag using lstat
	//
	memset(&statb, 0, sizeof(statb));
	assert(!(lstat(slink, &statb) < 0 ));
	assert(S_ISLNK(statb.st_mode));

	//
	// Test that readlink returns zero.
	//
	assert(!readlink(slink, &buf, 1));

	//
	// Delete test symlink, test directory and release all resources.
	//
	unlink(slink);
	unlink(parent_dir);
	free(slink);
	free(parent_dir);
	return 0;
}
