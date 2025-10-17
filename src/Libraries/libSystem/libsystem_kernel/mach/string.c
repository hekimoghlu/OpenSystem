/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 20, 2022.
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
#include "string.h"

static const char hex[] = "0123456789abcdef";

static int
_mach_strlen(const char *str)
{
	const char *p;
	for (p = str; p; p++) {
		if (*p == '\0') {
			return (int)(p - str);
		}
	}
	/* NOTREACHED */
	return 0;
}

static void
_mach_hex(char **buffer, int *length, unsigned long long n)
{
	char buf[32];
	char *cp = buf + sizeof(buf);

	if (n) {
		*--cp = '\0';
		while (n) {
			*--cp = hex[n & 0xf];
			n >>= 4;
		}

		int width = _mach_strlen(cp);
		while (width > 0 && length > 0) {
			*(*buffer)++ = *cp++;
			(*length)--;
			width--;
		}
	}
}

int
_mach_vsnprintf(char *buffer, int length, const char *fmt, va_list ap)
{
	int width, max = length;
	char *out_ptr = buffer;

	// we only ever write n-1 bytes so we can put a \0 at the end
	length--;
	while (length > 0 && *fmt) {
		if (*fmt == '\0') {
			break;
		}
		if (*fmt != '%') {
			*(out_ptr++) = *(fmt++);
			length--;
			continue;
		}
		fmt++;
		// only going to support a specific subset of sprintf flags
		// namely %s, %x, with no padding modifiers
		switch (*fmt++) {
		case 's':
		{
			char *cp = va_arg(ap, char*);
			width = _mach_strlen(cp);
			while (width > 0 && length > 0) {
				*(out_ptr++) = *(cp++);
				width--;
				length--;
			}
			break;
		}
		case 'x':
		{
			_mach_hex(&out_ptr, &length, va_arg(ap, unsigned int));
			break;
		}
		}
	}
	if (max > 0) {
		*out_ptr = '\0';
	}
	return max - (length + 1); /* don't include the final NULL in the return value */
}

int
_mach_snprintf(char *buffer, int length, const char *fmt, ...)
{
	int ret;
	va_list ap;
	va_start(ap, fmt);
	ret = _mach_vsnprintf(buffer, length, fmt, ap);
	va_end(ap);
	return ret;
}
