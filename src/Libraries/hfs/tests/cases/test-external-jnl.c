/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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
//  external-jnl.c
//  hfs
//
//  Created by Chris Suter on 8/11/15.
//
//

#include <TargetConditionals.h>

#if !TARGET_OS_IPHONE

#include <stdio.h>
#include <sys/mount.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>

#include "hfs-tests.h"
#include "disk-image.h"
#include "systemx.h"
#include "../core/hfs_format.h"
#include "test-utils.h"

#define HOST_IMAGE			"/tmp/external-jnl1.sparseimage"
#define EXTERNAL_IMAGE	"/tmp/external-jnl2.sparseimage"

TEST(external_jnl)

int run_external_jnl(__unused test_ctx_t *ctx)
{
	unlink(HOST_IMAGE);
	unlink(EXTERNAL_IMAGE);

	/* Since disk image cleanup occurs on a stack, create the external
	 * journal partition first so that the cleanup of the host image
	 * prevents a resource busy error during the journal partition ejection.
	 */
	disk_image_t *di_ext = disk_image_create(EXTERNAL_IMAGE,
							&(disk_image_opts_t){
								.partition_type = EXTJNL_CONTENT_TYPE_UUID,
								.size = 8 * 1024 * 1024
							});

	disk_image_t *di_host = disk_image_create(HOST_IMAGE,
							&(disk_image_opts_t){
								.size = 64 * 1024 * 1024
							});

	unmount(di_host->mount_point, 0);

	assert(!systemx("/sbin/newfs_hfs", SYSTEMX_QUIET, "-J", "-D", di_ext->disk, di_host->disk, NULL));

	assert(!systemx("/usr/sbin/diskutil", SYSTEMX_QUIET, "mount", di_host->disk, NULL));

	free((char *)di_host->mount_point);
	di_host->mount_point = NULL;

	struct statfs *mntbuf;
	int i, n = getmntinfo(&mntbuf, 0);
	for (i = 0; i < n; ++i) {
		if (!strcmp(mntbuf[i].f_mntfromname, di_host->disk)) {
			di_host->mount_point = strdup(mntbuf[i].f_mntonname);
			break;
		}
	}

	assert(i < n);

	char *path;
	asprintf(&path, "%s/test", di_host->mount_point);
	int fd = open(path, O_RDWR | O_CREAT, 0666);
	assert_with_errno(fd >= 0);
	assert_no_err(close(fd));
	assert_no_err(unlink(path));
	free(path);

	return 0;
}

#endif // !TARGET_OS_IPHONE
