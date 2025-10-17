/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 7, 2023.
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
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>

#include "hfs-tests.h"
#include "test-utils.h"
#include "disk-image.h"
#include "systemx.h"

TEST(readdir)

static disk_image_t *di;

int run_readdir(__unused test_ctx_t *ctx)
{
	di = disk_image_get();
	
	char *path;
	asprintf(&path, "%s/readdir.data", di->mount_point);

	systemx("/bin/rm", "-rf", path, NULL);
	assert_no_err(mkdir(path, 0777));
	
	char *file;
	asprintf(&file, "%s/file1", path);
	int fd;
	int numeof;

	assert_with_errno((fd = open(file, O_CREAT | O_TRUNC | O_RDWR, 0666)) >= 0);
	assert_no_err(close(fd));

	file[strlen(file) - 1] = '2';
	assert_with_errno((fd = open(file, O_CREAT | O_TRUNC | O_RDWR, 0666)) >= 0);
	assert_no_err(close(fd));

	DIR *dir;

	assert_with_errno((dir = opendir(path)) != NULL);

	struct dirent entry, *dp;
	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(!strcmp(dp->d_name, "."));
	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(!strcmp(dp->d_name, ".."));
	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(!strcmp(dp->d_name, "file1"));
	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(!strcmp(dp->d_name, "file2"));
	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(dp == NULL);

	closedir(dir);

	assert_with_errno((dir = opendir(path)) != NULL);

	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(!strcmp(dp->d_name, "."));
	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(!strcmp(dp->d_name, ".."));
	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(!strcmp(dp->d_name, "file1"));
	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(!strcmp(dp->d_name, "file2"));
	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(dp == NULL);
	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(dp == NULL);

	unlink(file);

	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(dp == NULL);

	assert_no_err(closedir(dir));

	assert_with_errno((dir = opendir(path)) != NULL);

	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(!strcmp(dp->d_name, "."));
	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(!strcmp(dp->d_name, ".."));
	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(!strcmp(dp->d_name, "file1"));
	assert_no_err(readdir_r(dir, &entry, &dp));
	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(dp == NULL);
	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(dp == NULL);

	file[strlen(file) - 1] = '1';
	unlink(file);

	assert_no_err(readdir_r(dir, &entry, &dp));
	assert(dp == NULL);

	assert_no_err(closedir(dir));

	assert_with_errno((dir = opendir(path)) != NULL);
	assert_with_errno((fd = open(file, O_CREAT | O_TRUNC | O_RDWR, 0666)) >= 0);
	assert_no_err(close(fd));

	numeof = 0;
	for (;;) {
		assert_no_err(readdir_r(dir, &entry, &dp));
		if (dp !=0)
			continue;
		assert_no_err(readdir_r(dir, &entry, &dp));
		if (++numeof == 3)
			break;
		if (numeof == 1)
			assert_no_err(unlink(file));

	}
	assert_no_err(closedir(dir));

	return 0;
}
