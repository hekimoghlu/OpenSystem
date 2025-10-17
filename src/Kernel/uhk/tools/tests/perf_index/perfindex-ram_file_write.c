/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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
#include "perf_index.h"
#include "fail.h"
#include "test_file_helper.h"
#include "ramdisk.h"
#include <sys/param.h>
#include <stdio.h>

const char ramdisk_name[] = "StressRAMDisk";
char ramdisk_path[MAXPATHLEN];

DECL_SETUP {
	int retval;

	retval = setup_ram_volume(ramdisk_name, ramdisk_path);
	VERIFY(retval == PERFINDEX_SUCCESS, "setup_ram_volume failed");

	printf("ramdisk: %s\n", ramdisk_path);

	return test_file_write_setup(ramdisk_path, num_threads, length);
}

DECL_TEST {
	return test_file_write(ramdisk_path, thread_id, num_threads, length, 0L);
}

DECL_CLEANUP {
	int retval;

	retval = test_file_write_cleanup(ramdisk_path, num_threads, length);
	VERIFY(retval == PERFINDEX_SUCCESS, "test_file_read_cleanup failed");

	retval = cleanup_ram_volume(ramdisk_path);
	VERIFY(retval == 0, "cleanup_ram_volume failed");

	return PERFINDEX_SUCCESS;
}
