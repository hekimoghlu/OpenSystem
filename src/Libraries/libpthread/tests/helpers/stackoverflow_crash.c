/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
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

#include<stdio.h>
#include<sys/resource.h>

static volatile int * array1_ref = NULL;
static long last_stack_addr = 0;

static void
recursive_fn(void)
{
	volatile int array1[1024]; /* leave this as it is */
	int addr;
	last_stack_addr = (long)&addr;
	array1_ref = array1; /* make sure compiler cannot discard storage */
	array1[0] = 0;
	if (array1_ref == 0) {
		/* fool clang -Winfinite-recursion */
		return;
	}
	recursive_fn();
	return;
}

int
main(__unused int argc, __unused const char *argv[])
{
	struct rlimit save;

	if (getrlimit(RLIMIT_STACK, &save) == -1) {
		printf("child: ERROR - getrlimit");
		return 2;
	}
	printf("child: LOG - current stack limits cur=0x%llx, max=0x%llx, inf=0x%llx\n", save.rlim_cur, save.rlim_max, RLIM_INFINITY);

	if(save.rlim_cur >= save.rlim_max) {
		printf("child: ERROR - invalid limits");
		return 2;
	}

	if(save.rlim_max == RLIM_INFINITY) {
		printf("child: ERROR - rlim_max = RLIM_INFINITY");
		return 2;
	}

	save.rlim_cur += 4;

	printf("child: LOG - Raising setrlimit rlim_cur=0x%llx, rlim_max=0x%llx\n", save.rlim_cur, save.rlim_max);

	if (setrlimit(RLIMIT_STACK, &save) == -1) {
		printf("child: ERROR - Raising the limits failed.");
		return 2;
	}

	printf("child: LOG - Make the stack grow such that a SIGSEGV is generated.\n");
	recursive_fn();
	return 0;
}
