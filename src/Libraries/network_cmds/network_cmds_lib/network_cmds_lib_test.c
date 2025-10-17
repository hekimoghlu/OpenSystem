/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <strings.h>
#include <sys/syslimits.h>

#include "network_cmds_lib.h"

uint8_t buffer[LINE_MAX];

static void
hexdump(void *data, size_t len)
{
	size_t i, j, k;
	unsigned char *ptr = (unsigned char *)data;
#define MAX_PER_LINE 16
#define HEX_SIZE (3 * MAX_PER_LINE)
#define ASCII_OFFSET (HEX_SIZE + 2)
#define ASCII_SIZE MAX_PER_LINE
#define BUF_SIZE (ASCII_OFFSET + MAX_PER_LINE + 1)

	for (i = 0; i < len; i += MAX_PER_LINE) {
		unsigned char buf[BUF_SIZE];

		memset(buf, ' ', BUF_SIZE);
		buf[BUF_SIZE - 1] = 0;

		for (j = i, k = 0; j < i + MAX_PER_LINE && j < len; j++) {
			unsigned char msnbl = ptr[j] >> 4;
			unsigned char lsnbl = ptr[j] & 0x0f;

			buf[k++] = msnbl < 10 ? msnbl + '0' : msnbl + 'a' - 10;
			buf[k++] = lsnbl < 10 ? lsnbl + '0' : lsnbl + 'a' - 10;

			buf[k++] = ' ';

			buf[ASCII_OFFSET + j - i] = isprint(ptr[j]) ? ptr[j] : '.';
		}
		(void) printf("%s\n", buf);
	}
}

int
main(int argc, char *argv[])
{
	char *str;

	for (uint16_t val = 0; val < UINT8_MAX; val++) {
		buffer[val] = (uint8_t)(val + 1);
	}
	buffer[UINT8_MAX] = 0;

	printf("\n# dirty string buffer:\n");
	hexdump(buffer, UINT8_MAX +1);

	str = (char *)buffer;
	str = clean_non_printable(str, UINT8_MAX);

	printf("\n# cleanup string buffer:\n");
	hexdump(str, UINT8_MAX +1);

	printf("\n# printf string:\n");
	printf("%s\n", str);

	return 0;
}
