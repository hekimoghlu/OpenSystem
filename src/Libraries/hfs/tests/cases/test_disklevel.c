/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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
#include <sys/attr.h>
#include <unistd.h>
#include <sys/stat.h>

#include "hfs-tests.h"
#include "test-utils.h"
#include "../core/hfs_fsctl.h"
#include "disk-image.h"

TEST(disklevel)

int run_disklevel(__unused test_ctx_t *ctx)
{
	disk_image_t *di = disk_image_get();
	const char *test_hfs_volume = di->mount_point;
	uint32_t very_low_disk = 0, low_disk = 0, near_low_disk = 0, desired_disk = 0;

	// Make sure initial values are sane.
	assert_no_err(fsctl(test_hfs_volume, HFSIOC_GET_VERY_LOW_DISK, &very_low_disk, 0));
	assert_no_err(fsctl(test_hfs_volume, HFSIOC_GET_LOW_DISK, &low_disk, 0));
	assert_no_err(fsctl(test_hfs_volume, APFSIOC_GET_NEAR_LOW_DISK, &near_low_disk, 0));
	assert_no_err(fsctl(test_hfs_volume, HFSIOC_GET_DESIRED_DISK, &desired_disk, 0));
	assert(very_low_disk > 0);
	assert(very_low_disk < low_disk);
	assert(low_disk < near_low_disk);
	assert(near_low_disk < desired_disk);

	very_low_disk = 1;
	low_disk = 2;
	near_low_disk = 3;
	desired_disk = 4;
	// Re-assign the values to new legal values and make sure they are preserved.
	assert_no_err(fsctl(test_hfs_volume, HFSIOC_SET_VERY_LOW_DISK, &very_low_disk, 0));
	assert_no_err(fsctl(test_hfs_volume, HFSIOC_SET_LOW_DISK, &low_disk, 0));
	assert_no_err(fsctl(test_hfs_volume, APFSIOC_SET_NEAR_LOW_DISK, &near_low_disk, 0));
	assert_no_err(fsctl(test_hfs_volume, HFSIOC_SET_DESIRED_DISK, &desired_disk, 0));

	assert_no_err(fsctl(test_hfs_volume, HFSIOC_GET_VERY_LOW_DISK, &very_low_disk, 0));
	assert_no_err(fsctl(test_hfs_volume, HFSIOC_GET_LOW_DISK, &low_disk, 0));
	assert_no_err(fsctl(test_hfs_volume, APFSIOC_GET_NEAR_LOW_DISK, &near_low_disk, 0));
	assert_no_err(fsctl(test_hfs_volume, HFSIOC_GET_DESIRED_DISK, &desired_disk, 0));
	assert_equal(very_low_disk, 1, "%d");
	assert_equal(low_disk, 2, "%d");
	assert_equal(near_low_disk, 3, "%d");
	assert_equal(desired_disk, 4, "%d");

	// Now, attempt to reassign the levels to illegal values and make sure they don't lose their previous value.
	very_low_disk = 4;
	low_disk = 1;
	near_low_disk = 2;
	desired_disk = 0;
	assert(fsctl(test_hfs_volume, HFSIOC_SET_VERY_LOW_DISK, &very_low_disk, 0) < 0);
	assert(fsctl(test_hfs_volume, HFSIOC_SET_LOW_DISK, &low_disk, 0) < 0);
	assert(fsctl(test_hfs_volume, APFSIOC_SET_NEAR_LOW_DISK, &near_low_disk, 0) < 0);
	assert(fsctl(test_hfs_volume, HFSIOC_SET_DESIRED_DISK, &desired_disk, 0) < 0);

	assert_no_err(fsctl(test_hfs_volume, HFSIOC_GET_VERY_LOW_DISK, &very_low_disk, 0));
	assert_no_err(fsctl(test_hfs_volume, HFSIOC_GET_LOW_DISK, &low_disk, 0));
	assert_no_err(fsctl(test_hfs_volume, APFSIOC_GET_NEAR_LOW_DISK, &near_low_disk, 0));
	assert_no_err(fsctl(test_hfs_volume, HFSIOC_GET_DESIRED_DISK, &desired_disk, 0));
	assert_equal(very_low_disk, 1, "%d");
	assert_equal(low_disk, 2, "%d");
	assert_equal(near_low_disk, 3, "%d");
	assert_equal(desired_disk, 4, "%d");

	return 0;
}
