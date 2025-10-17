/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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
#include "ramdisk.h"
#include "fail.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/param.h>

int
setup_ram_volume(const char* name, char* path)
{
	char *cmd;
	int retval;

	retval = asprintf(&cmd, "diskutil erasevolume HFS+ '%s' `hdiutil attach -nomount ram://1500000` >/dev/null", name);
	VERIFY(retval > 0, "asprintf failed");

	retval = system(cmd);
	VERIFY(retval == 0, "diskutil command failed");

	snprintf(path, MAXPATHLEN, "/Volumes/%s", name);

	free(cmd);

	return PERFINDEX_SUCCESS;
}

int
cleanup_ram_volume(char* path)
{
	char *cmd;
	int retval;

	retval = asprintf(&cmd, "umount -f '%s' >/dev/null", path);
	VERIFY(retval > 0, "asprintf failed");

	retval = system(cmd);
	VERIFY(retval == 0, "diskutil command failed");

	free(cmd);

	return PERFINDEX_SUCCESS;
}
