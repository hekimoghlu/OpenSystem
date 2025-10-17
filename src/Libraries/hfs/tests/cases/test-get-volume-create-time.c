/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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

#include "../../core/hfs_fsctl.h"
#include "hfs-tests.h"
#include "test-utils.h"
#include "disk-image.h"

/*
 * Just as a good measure we add this check so that compilation does
 * not break when compiled against older hfs_fsctl.h which did not
 * include HFSIOC_GET_VOL_CREATE_TIME.
 */
#if !defined(HFSIOC_GET_VOL_CREATE_TIME)
#define HFSIOC_GET_VOL_CREATE_TIME _IOR('h', 4, time_t)
#endif

TEST(get_volume_create_time)

int run_get_volume_create_time(__unused test_ctx_t *ctx)
{
	disk_image_t *di;
	time_t vol_create_time;

	di = disk_image_get();
	/*
	 * Volume create date is stored inside volume header in localtime.  The
	 * date is stored as 32-bit integer containing the number of seconds
	 * since midnight, January 1, 1904. We can safely assume that create
	 * date set for the volume will not be epoch.
	 */
	vol_create_time = 0;
	assert_no_err(fsctl(di->mount_point, HFSIOC_GET_VOL_CREATE_TIME,
		&vol_create_time, 0));
	if (!vol_create_time)
		assert_fail("fcntl HFSIOC_GET_VOL_CREATE_TIME failed to set "
			"volume create time.");
	return 0;
}
