/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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

#include <unistd.h>
#include <pthread.h>

#include "hfs-tests.h"
#include "test-utils.h"
#include "disk-image.h"
#include "systemx.h"

TEST(journal_toggle)

#define DMG "/tmp/journal-toggle.sparseimage"

static disk_image_t *di;
static volatile bool run = true;

void *thread1func(__unused void *arg)
{
	char *file;
	
	asprintf(&file, "%s/file", di->mount_point);
	
	while (run) {
		assert(!systemx("/usr/bin/touch", file, NULL));
		assert(!systemx("/bin/rm", file, NULL));
	}

	pthread_exit(NULL);
}

void *thread2func(__unused void *arg)
{
	while (run) {
		assert(!systemx("/usr/sbin/diskutil", SYSTEMX_QUIET, "disableJournal", di->mount_point, NULL));
		assert(!systemx("/usr/sbin/diskutil", SYSTEMX_QUIET, "enableJournal", di->mount_point, NULL));
	}

	pthread_exit(NULL);
}

int run_journal_toggle(__unused test_ctx_t *ctx)
{
	di = disk_image_create(DMG, &(disk_image_opts_t){
									.size = 32 * 1024 * 1024
								});

	pthread_t thread1, thread2;
	assert(!pthread_create(&thread1, NULL, thread1func, NULL));
	assert(!pthread_create(&thread2, NULL, thread2func, NULL));
	
	sleep(10);
	
	run = false;
	
	void *ret1, *ret2;
	assert(!pthread_join(thread1, &ret1));
	assert(!pthread_join(thread2, &ret2));
	
	return 0;
}

#endif // !TARGET_OS_IPHONE
