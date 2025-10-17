/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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

/* These are to make syscall.o shut up. */
int dry_run = 0;
int read_only = 1;
int list_only = 0;
int preserve_perms = 0;
int preserve_links = 0;
#ifdef EA_SUPPORT
int extended_attributes = 0;
#endif

int
main(int argc, char **argv)
{
	int i;

	if (argc <= 1) {
		fprintf(stderr, "trimslash: needs at least one argument\n");
		return 1;
	}

	for (i = 1; i < argc; i++) {
		trim_trailing_slashes(argv[i]);	/* modify in place */
		printf("%s\n", argv[i]);
	}
	return 0;
}
