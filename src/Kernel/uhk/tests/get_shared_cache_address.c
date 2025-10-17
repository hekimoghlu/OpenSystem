/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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
#include <sys/resource.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <libproc.h>
#include <mach-o/dyld.h>
#include <mach-o/dyld_priv.h>

/*
 * Test helper to retrieve the address of the shared cache. The helper
 * also verifies that the process is correctly marked to have the shared
 * cache reslid when interrogated through proc_pid_rusage()
 */
int
main(int argc, char **argv)
{
	size_t shared_cache_len = 0;
	struct rusage_info_v5 ru = {};

	if (argc != 2) {
		fprintf(stderr, "Invalid helper invocation");
		exit(1);
	}

	if (proc_pid_rusage(getpid(), RUSAGE_INFO_V5, (rusage_info_t *)&ru) != 0) {
		perror("proc_pid_rusage() helper");
		exit(1);
	}

	if (strcmp(argv[1], "check_rusage_flag") == 0) {
		if (!(ru.ri_flags & RU_PROC_RUNS_RESLIDE)) {
			fprintf(stderr, "Helper rusage flag check failed\n");
			exit(1);
		}
	}

	printf("%p\n", _dyld_get_shared_cache_range(&shared_cache_len));
	exit(0);
}
