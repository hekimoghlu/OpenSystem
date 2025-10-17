/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#include "internal.h"

#pragma mark -
#pragma mark Utility Functions

// libplatform does not have strstr() and we don't want to add any new
// dependencies on libc, so we have to implement a version of strntr()
// here. Fortunately, as it's only used to look for boot arguments, it does not
// have to be efficient. We can also assume that the source string is
// nul-terminated. Eventually, we will move the function to a more central
// location and use it to replace other uses of strstr().
const char *
malloc_common_strstr(const char *src, const char *target, size_t target_len)
{
	const char *next = src;
	while (*next) {
		if (!strncmp(next, target, target_len)) {
			return next;
		}
		next++;
	}
	return NULL;
}

// Converts a string to a long. If a non-numeric value is found, the
// return value is whatever has been accumulated so far. end_ptr always points
// to the character that caused the conversion to stop. We can't use strtol()
// etc because that would add a new dependency on libc. Eventually, this
// function could be made generally available within the library and used to
// replace the existing calls to strtol(). Currenly only handles non-negative
// numbers and does not detect overflow.
long
malloc_common_convert_to_long(const char *ptr, const char **end_ptr)
{
	long value = 0;
	while (*ptr) {
		char c = *ptr;
		if (c < '0' || c > '9') {
			break;
		}
		value = value * 10 + (c - '0');
		ptr++;
	}
	*end_ptr = ptr;
	return value;
}

// Looks for a sequence of the form "key=value" in the string 'src' and
// returns the location of the first character of 'value', or NULL if not
// found. No spaces are permitted around the "=".
const char *
malloc_common_value_for_key(const char *src, const char *key)
{
	const char *ptr = src;
	size_t keylen = strlen(key);
	while ((ptr = malloc_common_strstr(ptr, key, keylen)) != NULL) {
		ptr += keylen;
		if (*ptr == '=') {
			return ptr + 1;
		}
	}
	return NULL;
}

// Looks for a sequence of the form "key=value" in the string 'src' and
// returns the location of the first character of 'value'. No spaces are
// permitted around the "=". The value is copied to 'bufp', up to the first
// whitespace or nul character and bounded by maxlen, and nul-terminated.
// Returns bufp if the key was found, NULL if not.
const char *
malloc_common_value_for_key_copy(const char *src, const char *key,
							   char *bufp, size_t maxlen)
{
	const char *ptr = malloc_common_value_for_key(src, key);
	if (ptr) {
		char *to = bufp;
		while (maxlen > 1) { // Always leave room for a '\0'
			char c = *ptr++;
			if (c == '\0' || c == ' ' || c == '\t' || c == '\n') {
				break;
			}
			*to++ = c;
			maxlen--;
		}
		*to = '\0';	// Always nul-terminate
		return bufp;
	}
	return NULL;
}


