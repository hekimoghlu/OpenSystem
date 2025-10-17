/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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

//
//  symlink_test.c
//  copyfile_test
//

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <removefile.h>
#include <sys/fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "test_utils.h"

REGISTER_TEST(symlink, false, 30);

static bool test_symlink_copies(const char *test_directory, bool symlink_to_dir) {
	char src_name[BSIZE_B] = {0}, symlink_name[BSIZE_B] = {0};
	char dst_name[BSIZE_B] = {0}, dst_stored_path[BSIZE_B] = {0};
	struct stat dst_sb;
	int test_file_id;
	bool success = true;

	// Construct our source layout:
	//
	// apfs_test_directory
	//   testfile-NUM.symlink_src (directory if `symlink_to_dir`, else a regular file)
	//   testfile-NUM.symlink (symlink)
	//
	// We will copy into a destination in the same directory called
	//   testfile-NUM.symlink_dst
	// and verify that without COPYFILE_NOFOLLOW_SRC the destination is copied,
	// and with COPYFILE_NOFOLLOW_SRC the link itself is copied.
	//

	// Get ready for the test.
	test_file_id = rand () % DEFAULT_NAME_MOD;

	create_test_file_name(test_directory, "symlink_src", test_file_id, src_name);
	create_test_file_name(test_directory, "symlink", test_file_id, symlink_name);
	create_test_file_name(test_directory, "symlink_dst", test_file_id, dst_name);

	if (symlink_to_dir) {
		assert_no_err(mkdir(src_name, DEFAULT_MKDIR_PERM));
	} else {
		// Create a non-empty regular file and a symlink pointing to it.
		int fd = open(src_name, DEFAULT_OPEN_FLAGS, DEFAULT_OPEN_PERM);
		assert_with_errno(fd >= 0);
		assert_with_errno(write(fd, "j", 1) == 1);
		assert_no_err(close(fd));
		fd = -1;
	}

	assert_no_err(symlink(src_name, symlink_name));

	// Now, perform a copy of the symlink without COPYFILE_NOFOLLOW_SRC
	// and make sure the right thing happens.
	assert_no_err(copyfile(symlink_name, dst_name, NULL, COPYFILE_ALL));

	// Verify that the destination is actually the
	// regular file or directory we copied.
	assert_no_err(lstat(dst_name, &dst_sb));
	if (symlink_to_dir) {
		success = success && S_ISDIR(dst_sb.st_mode);
	} else {
		success = success && S_ISREG(dst_sb.st_mode);
		success = success && (dst_sb.st_size == 1);
		success = success && verify_copy_contents(src_name, dst_name);
	}

	// Now, repeat with COPYFILE_NOFOLLOW_SRC.
	assert_no_err(remove(dst_name));
	assert_no_err(copyfile(symlink_name, dst_name, NULL, COPYFILE_ALL|COPYFILE_NOFOLLOW_SRC));

	// Verify the destination is a symlink and points to the same place.
	assert_no_err(lstat(dst_name, &dst_sb));
	success = success && S_ISLNK(dst_sb.st_mode);
	assert_with_errno(readlink(dst_name, dst_stored_path, sizeof(dst_stored_path)) > 0);
	assert(!strcmp(src_name, dst_stored_path));

	assert_no_err(unlink(dst_name));

	// We're done!
	(void)remove(src_name);
	(void)remove(symlink_name);
	(void)remove(dst_name);

	return success;
}

bool do_symlink_test(const char *apfs_test_directory, __unused size_t block_size) {
	bool success = true;

	success = success && test_symlink_copies(apfs_test_directory, false);
	success = success && test_symlink_copies(apfs_test_directory, true);

	return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
