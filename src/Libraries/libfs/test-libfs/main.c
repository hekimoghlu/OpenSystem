/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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

#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include "test_harness.h"

#define NUM_TESTS 6

int test_simple_read(void) {
	return maketest("Deep down", 1, 0, 9);
}

int test_multiple_block_read(void) {
	return maketest("Deep down", 3, 0, 9);
}

int test_full_block_read(void) {
	return maketest("Deep", 4, 0, 4);
}

int test_offset_multiple_block_read(void) {
	return maketest("Louisiana", 3, 10, 9);
}

int test_offset_full_block_read(void) {
	return maketest("down Louis", 5, 5, 10);
}

int test_offset_multiple_full_block_read(void) {
	return maketest("ew Orl", 3, 30, 6);
}

int (*test_fns[NUM_TESTS])(void) =
{   test_simple_read,
	test_multiple_block_read,
	test_full_block_read,
	test_offset_multiple_block_read,
	test_offset_full_block_read,
	test_offset_multiple_full_block_read
};

int main(int argc, const char * argv[]) {
	int res;

	printf("Running tests...\n");
	for (int i = 0; i < NUM_TESTS; i++) {
		res = test_fns[i]();
		if (res < 0) {
			printf("[FAIL]: Test %d\n", i);
			return res;
		}
	}

	// If passed a device path, test our encryption status functions.
	if (argc > 1) {
		res = test_fs_encryption_status(argv[1]);
		if (res != 0) {
			printf("[FAIL]: Test %d\n", NUM_TESTS);
			return res;
		}

		res = test_di_encryption_status(argv[1]);
		if (res != 0) {
			printf("[FAIL]: Test %d\n", NUM_TESTS + 1);
			return res;
		}

		res = test_encryption_status(argv[1]);
		if (res != 0) {
			printf("[FAIL]: Test %d\n", NUM_TESTS + 2);
			return res;
		}
	}

	printf("[SUCCESS]\n");
	return 0;
}
