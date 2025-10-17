/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 23, 2021.
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
//  test-sparse-dev.c
//  hfs
//
//  Created by Chris Suter on 8/14/15.
//
//

#include <TargetConditionals.h>

#if !TARGET_OS_IPHONE

#include <sys/mount.h>
#include <sys/fcntl.h>
#include <unistd.h>

#include "hfs-tests.h"
#include "disk-image.h"
#include "test-utils.h"

TEST(sparse_dev)

int run_sparse_dev(__unused test_ctx_t *ctx)
{
	disk_image_t *di = disk_image_create("/tmp/sparse-dev.sparseimage",
										 &(disk_image_opts_t) {
											 .size = 64 * 1024 * 1024
										 });

	char *path;
	asprintf(&path, "%s/child.sparseimage", di->mount_point);

	disk_image_t *child = disk_image_create(path,
											&(disk_image_opts_t) {
													.size = 256 * 1024 * 1024
											});

	free(path);

	asprintf(&path, "%s/test.file", child->mount_point);

	int fd = open(path, O_CREAT | O_RDWR, 0777);
	assert_with_errno(fd >= 0);

	assert(ftruncate(fd, 128 * 1024 * 1024) == -1 && errno == ENOSPC);

	struct statfs sfs;
	assert_no_err(statfs(child->mount_point, &sfs));

	assert(sfs.f_bfree * sfs.f_bsize < 64 * 1024 * 1024);


	assert_no_err (close(fd));

	return 0;
}

#endif // !TARGET_OS_IPHONE
