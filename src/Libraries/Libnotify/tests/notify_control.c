/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 21, 2024.
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
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <notify_private.h>

int main(int argc, char *argv[])
{
	pid_t pid;
	int i;
	uint32_t status;

	for (i = 1; i < argc; i++)
	{
		if (!strcmp(argv[i], "-s"))
		{
			pid = atoi(argv[++i]);
			status = notify_suspend_pid(pid);
			if (status != 0) printf("suspend pid %d failed status %u\n", pid, status);
			else printf("suspend pid %d OK\n", pid);
		}
		else if (!strcmp(argv[i], "-r"))
		{
			pid = atoi(argv[++i]);
			status = notify_resume_pid(pid);
			if (status != 0) printf("resume pid %d failed status %u\n", pid, status);
			else printf("resume pid %d OK\n", pid);
		}
	}

	return 0;
}
