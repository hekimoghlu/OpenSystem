/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 14, 2023.
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

#include <assert.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>

#include "../md4.h"
#include "../extern.h"

int
main(int argc, char *argv[])
{
	int	 	 fd;
	struct opts	 opts;
	size_t		 sz;
	struct flist	*fl;

	memset(&opts, 0, sizeof(struct opts));

	assert(2 == argc);

	fd = open(argv[1], O_NONBLOCK | O_RDONLY, 0);
	assert(fd != -1);

	fl = flist_recv(&opts, fd, &sz);
	flist_free(fl, sz);
	return EXIT_SUCCESS;
}
