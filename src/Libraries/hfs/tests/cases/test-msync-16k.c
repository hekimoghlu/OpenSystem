/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/errno.h>
#include <sys/stat.h>
#include <stdio.h>
#include <sched.h>

#include "hfs-tests.h"
#include "test-utils.h"
#include "disk-image.h"

TEST(msync_16k)

static disk_image_t *di;

volatile off_t size = 0;

static int fd;

void syncer(void)
{
	for (;;) {
		off_t sz = size;

		if (sz == -1)
			break;

		sz &= ~0xfffff;

		void *p = mmap(NULL, 0x100000, PROT_READ | PROT_WRITE, MAP_SHARED,
					   fd, sz);

		assert_with_errno(p != MAP_FAILED);

		while ((size & ~0xfffff) == sz) {
			assert_no_err(msync(p, 0x100000, MS_SYNC));
			sched_yield();
		}

		assert_no_err(munmap(p, 0x100000));
	}
}

int run_msync_16k(__unused test_ctx_t *ctx)
{
	di = disk_image_get();
	
	char *file;
	asprintf(&file, "%s/msync-16k.data", di->mount_point);
	
	char *buf = malloc(1024 * 1024);
	memset(buf, 0xaf, 1024 * 1024);

	unlink(file);

	fd = open(file, O_CREAT | O_RDWR, 0666);
	assert_with_errno(fd >= 0);

	pthread_t thr;
	pthread_create(&thr, NULL, (void *(*)(void *))syncer, NULL);

	assert_no_err(fcntl(fd, F_NOCACHE, 1));

	check_io(write(fd, buf, 8192), 8192);

	size = 8192;
	while (size < 100 * 1024 * 1024ll) {
		// Force an extent boundary
		struct log2phys l2p = {
			.l2p_contigbytes = 4096,
			.l2p_devoffset = size - 4096,
		};
		assert(!fcntl(fd, F_LOG2PHYS_EXT, &l2p));

		struct fstore fst = {
			.fst_posmode = F_VOLPOSMODE,
			.fst_offset = l2p.l2p_devoffset + 4096,
			.fst_length = size + 16384,
		};
		assert(!fcntl(fd, F_PREALLOCATE, &fst));

 		check_io(pwrite(fd, buf, 16384, size), 16384);
		size += 16384;
	}

	size_t sz = size;
	size = -1;

	pthread_join(thr, NULL);
	assert_no_err(close(fd));

	fd = open(file, O_RDWR);

	assert_with_errno(fd >= 0);
	size_t done = 0;
	char *cmp_buf = malloc(1024 * 1024);

	while (done < sz) {
		void *p = mmap(NULL, 1024 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, fd, done);
		assert_with_errno(p != MAP_FAILED);
		assert_no_err(msync(p, 1024 * 1024, MS_INVALIDATE | MS_SYNC));
		assert_no_err(munmap(p, 1024 * 1024));

		bzero(cmp_buf, 1024 * 1024);
		ssize_t amt = read(fd, cmp_buf, 1024 * 1024);
		assert(amt > 0);
		assert(!memcmp(cmp_buf, buf, amt));
		done += amt;
	}

	assert_no_err(close(fd));

	assert_no_err(unlink(file));
	
	free(cmp_buf);
	free(buf);
	free(file);

	return 0;
}
