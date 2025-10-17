/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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
/*
 * Mach Operating System
 * Copyright (c) 1991,1990,1989 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS"
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie Mellon
 * the rights to redistribute these changes.
 */
/*
 * mig_strncpy.c - by Joshua Block
 *
 * mig_strncpy -- Bounded string copy.  Does what the library routine strncpy
 * OUGHT to do:  Copies the (null terminated) string in src into dest, a
 * buffer of length len.  Assures that the copy is still null terminated
 * and doesn't overflow the buffer, truncating the copy if necessary.
 *
 * Parameters:
 *
 *     dest - Pointer to destination buffer.
 *
 *     src - Pointer to source string.
 *
 *     len - Length of destination buffer.
 *
 * Result:
 *	length of string copied, INCLUDING the trailing 0.
 */
#include <mach/mig_errors.h>

int
mig_strncpy(
	char *dest,
	const char *src,
	int len)
{
	int i;

	if (len <= 0) {
		return 0;
	}

	for (i = 1; i < len; i++) {
		if (!(*dest++ = *src++)) {
			return i;
		}
	}

	*dest = '\0';
	return i;
}

/*
 * mig_strncpy_zerofill -- Bounded string copy.  Does what the
 * library routine strncpy OUGHT to do:  Copies the (null terminated)
 * string in src into dest, a buffer of length len.  Assures that
 * the copy is still null terminated and doesn't overflow the buffer,
 * truncating the copy if necessary. If the string in src is smaller
 * than given length len, it will zero fill the remaining bytes in dest.
 *
 * Parameters:
 *
 *     dest - Pointer to destination buffer.
 *
 *     src - Pointer to source string.
 *
 *     len - Length of destination buffer.
 *
 * Result:
 *	length of string copied, INCLUDING the trailing 0.
 */
int
mig_strncpy_zerofill(
	char *dest,
	const char *src,
	int len)
{
	int i;
	boolean_t terminated = FALSE;
	int retval = 0;

	if (len <= 0 || dest == 0) {
		return 0;
	}

	if (src == 0) {
		terminated = TRUE;
	}

	for (i = 1; i < len; i++) {
		if (!terminated) {
			if (!(*dest++ = *src++)) {
				retval = i;
				terminated = TRUE;
			}
		} else {
			*dest++ = '\0';
		}
	}

	*dest = '\0';
	if (!terminated) {
		retval = i;
	}

	return retval;
}
