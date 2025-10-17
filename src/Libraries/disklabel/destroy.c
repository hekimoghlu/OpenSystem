/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <err.h>
#include <unistd.h>

#include "util.h"

void
doDestroy(const char *dev) {
	int bs;
	if (gVerbose) {
		fprintf(stderr, "Destroying device %s", dev);
	}

	if (!IsAppleLabel(dev)) {
		errx(4,"doDestroy:  device %s is not an Apple Label device", dev);
	}

	bs = GetBlockSize(dev);
	if (bs != 0) {
		int fd;
		char bz[bs];
		memset(bz, 0, bs);

		fd = open(dev, O_WRONLY);
		if (fd == -1) {
			err(1, "doDestroy:  cannot open device %s for writing", dev);
		}
		if (write(fd, bz, bs) != bs) {
			err(2, "doDestroy:  cannot write %d bytes onto device %s", bs, dev);
		}
		close(fd);
	} else {
		errx(3, "doDestroy:  cannot get blocksize for device %s", dev);
	}
}
