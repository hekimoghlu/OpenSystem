/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 21, 2025.
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

#if !TARGET_OS_IPHONE
 
#include <sys/ioctl.h>
#include <sys/ioccom.h>
#include <sys/param.h>
#include <sys/mount.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/errno.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>

#include "../../core/hfs_fsctl.h"

#include "hfs-tests.h"
#include "test-utils.h"
#include "systemx.h"
#include "disk-image.h"

TEST(scan_range_size, .run_as_root = true)

static disk_image_t *di;

static hfs_fsinfo	fsinfo;

static void test_fsinfo_file_extent_size(void)
{
	bzero(&fsinfo, sizeof(fsinfo));
	fsinfo.header.request_type = HFS_FSINFO_FILE_EXTENT_SIZE;
	fsinfo.header.version = HFS_FSINFO_VERSION;
	assert_no_err(fsctl(di->mount_point, HFSIOC_GET_FSINFO, &fsinfo, 0));
}

static void test_fsinfo_free_extents(void)
{
	bzero(&fsinfo, sizeof(fsinfo));
	fsinfo.header.version = HFS_FSINFO_VERSION;
	fsinfo.header.request_type = HFS_FSINFO_FREE_EXTENTS;
	assert_no_err(fsctl(di->mount_point, HFSIOC_GET_FSINFO, &fsinfo, 0));
}


int run_scan_range_size(__unused test_ctx_t *ctx) {

	di = disk_image_get();

	test_fsinfo_file_extent_size();
	test_fsinfo_free_extents();

	return 0;
}

#endif // !TARGET_OS_IPHONE
