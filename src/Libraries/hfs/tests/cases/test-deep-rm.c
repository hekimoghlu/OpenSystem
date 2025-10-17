/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 8, 2025.
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
#include <spawn.h>

#include "hfs-tests.h"
#include "test-utils.h"
#include "disk-image.h"

TEST(deep_rm)

static disk_image_t *di;

static void rm_all(const char *path)
{
	char *p = strdup(path);

	pid_t pid;
	assert_no_err(posix_spawn(&pid, "/bin/rm", NULL, NULL,
							  (char *[]){ "rm", "-rf", p, NULL }, NULL));

	free(p);

	int stat;
	assert_with_errno(waitpid(pid, &stat, 0) == pid);
}

int run_deep_rm(__unused test_ctx_t *ctx)
{
	di = disk_image_get();
	
	char *dir;
	asprintf(&dir, "%s/deep-rm-test", di->mount_point);
	
	rm_all(dir);

	assert_no_err(mkdir(dir, 0777));

	char path[4096];
	strcpy(path, dir);

	char *end = path + strlen(path);

	for (int i = 0; i < 100; ++i) {
		memcpy(end, "/dir", 5);
		assert_no_err(mkdir(path, 0777));
		end += 4;
	}

	assert_no_err(chdir(path));

	rm_all(dir);

	assert_no_err(chdir(di->mount_point));
	free(dir);

	return 0;
}
