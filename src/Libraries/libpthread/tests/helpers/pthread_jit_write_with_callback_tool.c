/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 19, 2022.
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include <pthread.h>
#include <dlfcn.h>
#include <TargetConditionals.h>

static int
test_callback_should_not_actually_run(void *ctx)
{
	(void)ctx;
	printf("This callback was not expected to actually run\n");
	return 0;
}

#define HELPER_LIBRARY_PATH "/AppleInternal/Tests/libpthread/assets/libpthreadjittest.notdylib"

int
main(int argc, char *argv[])
{
	if (argc != 2) {
		// The test is checking whether we exited with a signal to see if the
		// expected abort occurs, so if we need to bail out because of a
		// misconfiguration we should try to exit with an error code instead
		return 1;
	}
#if TARGET_OS_OSX
	if (!strcmp(argv[1], "pthread_jit_write_protect_np")) {
		printf("Attempting pthread_jit_write_protect_np\n");
		pthread_jit_write_protect_np(false);
		printf("Should not have made it here\n");
	} else
#endif // TARGET_OS_OSX
	if (!strcmp(argv[1], "pthread_jit_write_with_callback_np")) {
		printf("Attempting pthread_jit_write_with_callback_np\n");
		(void)pthread_jit_write_with_callback_np(
				test_callback_should_not_actually_run, NULL);
		printf("Should not have made it here\n");
	} else if (!strcmp(argv[1], "pthread_jit_write_freeze_callbacks_np")) {
		printf("Attempting freeze + dlopen + write_with_callback\n");

		pthread_jit_write_freeze_callbacks_np();

		void *handle = dlopen(HELPER_LIBRARY_PATH, RTLD_NOW);
		if (!handle) {
			printf("dlopen failed\n");
			return 1;
		}

		pthread_jit_write_callback_t cb = dlsym(handle,
				"test_dylib_jit_write_callback");
		if (!cb) {
			printf("dlsym failed\n");
			return 1;
		}

		(void)pthread_jit_write_with_callback_np(cb, NULL);
		printf("Should not have made it here\n");
	}

	return 1;
}
