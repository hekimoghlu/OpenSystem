/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD$");

#include <sys/types.h>
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <libutil.h>
#include <stdint.h>

int
expand_number(const char *buf, uint64_t *num)
{
	char *endptr;
	uintmax_t umaxval;
	uint64_t number;
	unsigned shift;
	int serrno;

	serrno = errno;
	errno = 0;
	umaxval = strtoumax(buf, &endptr, 0);
	if (umaxval > UINT64_MAX)
		errno = ERANGE;
	if (errno != 0)
		return (-1);
	errno = serrno;
	number = umaxval;

	switch (tolower((unsigned char)*endptr)) {
	case 'e':
		shift = 60;
		break;
	case 'p':
		shift = 50;
		break;
	case 't':
		shift = 40;
		break;
	case 'g':
		shift = 30;
		break;
	case 'm':
		shift = 20;
		break;
	case 'k':
		shift = 10;
		break;
	case 'b':
	case '\0': /* No unit. */
		*num = number;
		return (0);
	default:
		/* Unrecognized unit. */
		errno = EINVAL;
		return (-1);
	}

	if ((number << shift) >> shift != number) {
		/* Overflow */
		errno = ERANGE;
		return (-1);
	}
	*num = number << shift;
	return (0);
}
