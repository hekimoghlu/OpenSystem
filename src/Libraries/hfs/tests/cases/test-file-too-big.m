/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 2, 2024.
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

#include <TargetConditionals.h>

#if !(TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR)

#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <spawn.h>

#import <Foundation/Foundation.h>

#include "hfs-tests.h"
#include "test-utils.h"
#include "disk-image.h"

TEST(file_too_big)

#define DISK_IMAGE "/tmp/file_too_big.sparseimage"

int run_file_too_big(__unused test_ctx_t *ctx)
{
	int fd;
	
	disk_image_t *di = disk_image_create(DISK_IMAGE, &(disk_image_opts_t){
														.size = 16 TB
													});
	
	struct timeval start, end, elapsed;

	char *file;
	asprintf(&file, "%s/file-too-big", di->mount_point);
	assert_with_errno((fd = open(file, O_RDWR | O_CREAT, 0666)) >= 0);

	assert_no_err(gettimeofday(&start, NULL));

	assert(pwrite(fd, &fd, 4, 128 * 1024ull * 1024 * 1024 * 1024) == -1
		   && errno == ENOSPC);

	assert_no_err(close(fd));

	assert_no_err(gettimeofday(&end, NULL));

	timersub(&end, &start, &elapsed);
	assert(elapsed.tv_sec < 1);

	// Check truncate

	assert_no_err(gettimeofday(&start, NULL));

	assert(truncate(file, 128 * 1024ull * 1024 * 1024 * 1024) == -1);
	assert_with_errno(errno == ENOSPC);

	assert_no_err(gettimeofday(&end, NULL));
	
	timersub(&end, &start, &elapsed);
	assert(elapsed.tv_sec < 1);

	// Check preallocate

	assert_no_err(gettimeofday(&start, NULL));

	assert((fd = open(file, O_RDWR)) >= 0);

	fstore_t fst = {
		.fst_flags = F_ALLOCATEALL,
		.fst_posmode = F_PEOFPOSMODE,
		.fst_length = 128 * 1024ull * 1024 * 1024 * 1024,
	};

	assert(fcntl(fd, F_PREALLOCATE, &fst) == -1 && errno == ENOSPC);
	assert(fst.fst_bytesalloc == 0);

	assert_no_err(close(fd));

	assert_no_err(gettimeofday(&end, NULL));

	timersub(&end, &start, &elapsed);
	assert(elapsed.tv_sec < 1);

	// And check preallocate works without the F_ALLOCATEALL flag

	assert((fd = open(file, O_RDWR)) >= 0);

	fst.fst_flags = 0;

	assert(fcntl(fd, F_PREALLOCATE, &fst) == -1 && errno == ENOSPC);

	// It should have allocated at least 32 MB
	assert(fst.fst_bytesalloc > 32 * 1024 * 1024);

	assert_no_err(close(fd));

	return 0;
}

#endif // !(TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR)
