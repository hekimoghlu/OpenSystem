/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 1, 2024.
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

int
main(UNUSED(int argc), UNUSED(char *argv[]))
{
	int n, i;
	gid_t *list;
	gid_t gid = MY_GID();
	int gid_in_list = 0;

#ifdef HAVE_GETGROUPS
	if ((n = getgroups(0, NULL)) < 0) {
		perror("getgroups");
		return 1;
	}
#else
	n = 0;
#endif

	list = (gid_t*)malloc(sizeof (gid_t) * (n + 1));
	if (!list) {
		fprintf(stderr, "out of memory!\n");
		exit(1);
	}

#ifdef HAVE_GETGROUPS
	if (n > 0)
		n = getgroups(n, list);
#endif

	for (i = 0; i < n; i++)  {
		printf("%lu ", (unsigned long)list[i]);
		if (list[i] == gid)
			gid_in_list = 1;
	}
	/* The default gid might not be in the list on some systems. */
	if (!gid_in_list)
		printf("%lu", (unsigned long)gid);
	printf("\n");

	return 0;
}
