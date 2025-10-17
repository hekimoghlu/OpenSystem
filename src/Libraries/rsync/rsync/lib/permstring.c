/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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
#include "rsync.h"

/* Produce a string representation of Unix mode bits like that used by ls(1).
 * The "buf" buffer must be at least 11 characters. */
void permstring(char *perms, mode_t mode)
{
	static const char *perm_map = "rwxrwxrwx";
	int i;

	strlcpy(perms, "----------", 11);

	for (i = 0; i < 9; i++) {
		if (mode & (1 << i))
			perms[9-i] = perm_map[8-i];
	}

	/* Handle setuid/sticky bits.  You might think the indices are
	 * off by one, but remember there's a type char at the
	 * start.  */
	if (mode & S_ISUID)
		perms[3] = (mode & S_IXUSR) ? 's' : 'S';

	if (mode & S_ISGID)
		perms[6] = (mode & S_IXGRP) ? 's' : 'S';

#ifdef S_ISVTX
	if (mode & S_ISVTX)
		perms[9] = (mode & S_IXOTH) ? 't' : 'T';
#endif

	if (S_ISDIR(mode))
		perms[0] = 'd';
	else if (S_ISLNK(mode))
		perms[0] = 'l';
	else if (S_ISBLK(mode))
		perms[0] = 'b';
	else if (S_ISCHR(mode))
		perms[0] = 'c';
	else if (S_ISSOCK(mode))
		perms[0] = 's';
	else if (S_ISFIFO(mode))
		perms[0] = 'p';
}
