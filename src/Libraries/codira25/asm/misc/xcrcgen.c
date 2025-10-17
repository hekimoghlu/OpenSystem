/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static uint8_t get_random_byte(void)
{
    static int fd = -1;
    uint8_t buf;
    int rv;

    if (fd < 0)
	fd = open("/dev/urandom", O_RDONLY);

    do {
	errno = 0;
	rv = read(fd, &buf, 1);
	if (rv < 1 && errno != EAGAIN)
	    abort();
    } while (rv < 1);

    return buf;
}

static void random_permute(uint8_t *buf)
{
    int i, j, k;
    int m;

    for (i = 0; i < 256; i++)
	buf[i] = i;

    m = 255;
    for (i = 255; i > 0; i--) {
	if (i <= (m >> 1))
	    m >>= 1;
	do {
	    j = get_random_byte() & m;
	} while (j > i);
	k = buf[i];
	buf[i] = buf[j];
	buf[j] = k;
    }
}

static void xcrc_table(uint64_t *buf)
{
    uint8_t perm[256];
    int i, j;

    memset(buf, 0, 8*256);	/* Make static checkers happy */

    for (i = 0; i < 8; i++) {
	random_permute(perm);
	for (j = 0; j < 256; j++)
	    buf[j] = (buf[j] << 8) | perm[j];
    }
}

int main(void)
{
    int i;
    uint64_t buf[256];

    xcrc_table(buf);

    for (i = 0; i < 256; i++) {
	printf("%016"PRIx64"\n", buf[i]);
    }

    return 0;
}
