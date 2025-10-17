/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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

/****************************************************************************
simple routine to do connection counting
****************************************************************************/
int claim_connection(char *fname,int max_connections)
{
	int fd, i;

	if (max_connections <= 0)
		return 1;

	fd = open(fname,O_RDWR|O_CREAT, 0600);

	if (fd == -1) {
		return 0;
	}

	/* find a free spot */
	for (i=0;i<max_connections;i++) {
		if (lock_range(fd, i*4, 4)) return 1;
	}

	/* only interested in open failures */
	errno = 0;

	close(fd);
	return 0;
}
