/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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
#include <ctype.h>

#define MAXBUF 16

void
hex_and_ascii_print(const char *prefix, const void *data,
					size_t len, const char *suffix)
{
	size_t i, j, k;
	unsigned char *ptr = (unsigned char *)data;
	unsigned char hexbuf[3 * MAXBUF + 1];
	unsigned char asciibuf[MAXBUF + 1];
	
	for (i = 0; i < len; i += MAXBUF) {
		for (j = i, k = 0; j < i + MAXBUF && j < len; j++) {
			unsigned char msnbl = ptr[j] >> 4;
			unsigned char lsnbl = ptr[j] & 0x0f;
			
			if (isprint(ptr[j]))
				asciibuf[j % MAXBUF]  = ptr[j];
			else
				asciibuf[j % MAXBUF]  = '.';
			asciibuf[(j % MAXBUF) + 1]  = 0;
			
			hexbuf[k++] = msnbl < 10 ? msnbl + '0' : msnbl + 'a' - 10;
			hexbuf[k++] = lsnbl < 10 ? lsnbl + '0' : lsnbl + 'a' - 10;
			if ((j % 2) == 1)
				hexbuf[k++] = ' ';
		}
		for (; j < i + MAXBUF;j++) {
			hexbuf[k++] = ' ';
			hexbuf[k++] = ' ';
			if ((j % 2) == 1)
				hexbuf[k++] = ' ';
		}
		hexbuf[k] = 0;
		printf("%s%s  %s%s", prefix, hexbuf, asciibuf, suffix);
	}
}
