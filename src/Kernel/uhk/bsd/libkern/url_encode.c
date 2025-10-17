/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 19, 2023.
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
#include <libkern/libkern.h>

static int
hex2int(int c)
{
	if (c >= '0' && c <= '9') {
		return c - '0';
	} else if (c >= 'A' && c <= 'F') {
		return c - 'A' + 10;
	} else if (c >= 'a' && c <= 'f') {
		return c - 'a' + 10;
	}
	return 0;
}

static bool
isprint(int ch)
{
	return ch >= 0x20 && ch <= 0x7e;
}

/*
 * In-place decode of URL percent-encoded str
 */
void
url_decode(char *str)
{
	if (!str) {
		return;
	}

	while (*str) {
		if (*str == '%') {
			char c = 0;
			char *esc = str++; /* remember the start of the escape sequence */

			if (*str) {
				c += hex2int(*str++);
			}
			if (*str) {
				c = (char)((c << 4) + hex2int(*str++));
			}

			if (isprint(c)) {
				/* overwrite the '%' with the new char, and bump the rest of the
				 * string down a few characters */
				*esc++ = c;
				str = memmove(esc, str, strlen(str) + 1);
			}
		} else {
			str++;
		}
	}
}
