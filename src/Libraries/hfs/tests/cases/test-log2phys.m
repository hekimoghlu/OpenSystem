/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 12, 2022.
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
#include <limits.h>
#include <unistd.h>
#include <spawn.h>
#include <sys/stat.h>
#include <TargetConditionals.h>

#import <Foundation/Foundation.h>

#include "hfs-tests.h"
#include "test-utils.h"
#include "disk-image.h"

TEST(log2phys)

static disk_image_t *di;

int run_log2phys(__unused test_ctx_t *ctx)
{
	di = disk_image_get();
	char *file;
	asprintf(&file, "%s/log2phys.data", di->mount_point);
	
	int fd = open(file, O_RDWR | O_CREAT, 0666);

	struct log2phys l2p = {
		.l2p_contigbytes = OFF_MAX,
	};

	assert_no_err(ftruncate(fd, 1000));
	assert_no_err(fcntl(fd, F_LOG2PHYS_EXT, &l2p));

	l2p.l2p_contigbytes = -1;
	assert_with_errno(fcntl(fd, F_LOG2PHYS_EXT, &l2p) == -1 && errno == EINVAL);

	assert_no_err(close(fd));
	free(file);

	return 0;
}
