/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 22, 2022.
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
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/sysctl.h>
#include <ptrauth.h>
#include <math.h>
#include <string.h>

__attribute__((noinline))
static void
foo(void)
{
	printf("In foo()\n");
	fflush(stdout);
}

/*
 * volatile to stop the compiler from optimizing away calls to atan()
 */
volatile double zero = 0.0;

int
main(int argc, char **argv)
{
	void *addr;
	size_t s = sizeof(addr);
	int err;
	int a;

	/*
	 * needs to run as root for sysctl.
	 */
	if (geteuid() != 0) {
		printf("Test not running as root\n");
		exit(-1);
	}

	if (strcmp(argv[argc - 1], "foo") == 0) {
		foo();
	} else if (strcmp(argv[argc - 1], "Xfoo") == 0) {
		printf("Warm up call to foo()\n");
		foo();
		addr = ptrauth_strip(&foo, ptrauth_key_function_pointer);
		err = sysctlbyname("vm.corrupt_text_addr", NULL, NULL, &addr, s);
		foo();
	} else if (strcmp(argv[argc - 1], "atan") == 0) {
		printf("atan(0) is %g\n", atan(zero));
	} else if (strcmp(argv[argc - 1], "Xatan") == 0) {
		printf("Warmup call to atan(0) is %g\n", atan(zero));
		addr = ptrauth_strip(&atan, ptrauth_key_function_pointer);
		err = sysctlbyname("vm.corrupt_text_addr", NULL, NULL, &addr, s);
		printf("atan(0) is %g\n", atan(zero));
	} else {
		exit(-1);
	}
}
