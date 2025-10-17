/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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
#include <sys/time.h>

#include <err.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>

int
main(int argc, char *argv[])
{
	struct timeval times[2];
	unsigned long long val;

	if (argc < 2)
		errx(1, "usage: %s file [file ...]", getprogname());

	/* Overflows localtime() - used for testing ls(1) resilience */
	val = 67768036191705600;

	times[0].tv_sec = (time_t)val;
	times[0].tv_usec = 0;
	times[1] = times[0];

	for (int i = 1; i < argc; i++) {
		if (utimes(argv[i], times) != 0)
			err(1, "utimes(%s)", argv[i]);
	}

	return (0);
}
