/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 15, 2023.
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
#include "test_fault_helper.h"
#include "fail.h"
#include <sys/mman.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <TargetConditionals.h>

#if (TARGET_OS_IPHONE && !TARGET_OS_SIMULATOR)
#define MEMSIZE (1L<<28)
#else
#define MEMSIZE (1L<<30)
#endif

static char* memblock;

int
test_fault_setup()
{
	char *ptr;
	int pgsz = getpagesize();
	int retval;

	memblock = (char *)mmap(NULL, MEMSIZE, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
	VERIFY(memblock != MAP_FAILED, "mmap failed");

	/* make sure memory is paged */
	for (ptr = memblock; ptr < memblock + MEMSIZE; ptr += pgsz) {
		*ptr = 1;
	}

	/* set to read only, then back to read write so it faults on first write */
	retval = mprotect(memblock, MEMSIZE, PROT_READ);
	VERIFY(retval == 0, "mprotect failed");

	retval = mprotect(memblock, MEMSIZE, PROT_READ | PROT_WRITE);
	VERIFY(retval == 0, "mprotect failed");

	return PERFINDEX_SUCCESS;
}

int
test_fault_helper(int thread_id, int num_threads, long long length, testtype_t testtype)
{
	char *ptr;
	int pgsz = getpagesize();
	int retval;

	long long num_pages = MEMSIZE / pgsz;
	long long region_len = num_pages / num_threads;
	long long region_start = region_len * thread_id;
	long long region_end;

	if (thread_id < num_pages % num_threads) {
		region_start += thread_id;
		region_len++;
	} else {
		region_start += num_pages % num_threads;
	}

	region_start *= pgsz;
	region_len *= pgsz;
	region_end = region_start + region_len;

	long long left = length;

	while (1) {
		for (ptr = memblock + region_start; ptr < memblock + region_end; ptr += pgsz) {
			*ptr = 1;
			left--;
			if (left == 0) {
				break;
			}
		}

		if (left == 0) {
			break;
		}

		if (testtype == TESTFAULT) {
			retval = mprotect(memblock + region_start, region_len, PROT_READ) == 0;
			VERIFY(retval == 0, "mprotect failed");
			retval = mprotect(memblock + region_start, region_len, PROT_READ | PROT_WRITE) == 0;
			VERIFY(retval == 0, "mprotect failed");
		} else if (testtype == TESTZFOD) {
			retval = munmap(memblock + region_start, region_len) == 0;
			VERIFY(retval == 0, "munmap failed");
			ptr = mmap(memblock + region_start, region_len, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE | MAP_FIXED, -1, 0);
			VERIFY(ptr != 0, "mmap failed");
		}
	}
	return PERFINDEX_SUCCESS;
}
