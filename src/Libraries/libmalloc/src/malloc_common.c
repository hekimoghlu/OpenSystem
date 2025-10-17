/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 13, 2025.
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
// dependencies on libc, so we have to implement a version of strstr()
// here. Fortunately, as it's only used to look for boot arguments, it does not
// have to be efficient. We can also assume that the source string is
// nul-terminated. Eventually, we will move the function to a more central
// location and use it to replace other uses of strstr().
const char * __null_terminated
malloc_common_strstr(const char * __null_terminated src, const char * __counted_by(target_len) target, size_t target_len)
{
#if !MALLOC_TARGET_EXCLAVES
	const char *next = src;
	while (*next) {
		if (!strncmp(next, target, target_len)) {
			return next;
		}
		next++;
	}
	return NULL;
#else
	return strstr(src, __unsafe_null_terminated_from_indexable(target, target + target_len));
#endif // MALLOC_TARGET_EXCLAVES
}

// Converts a string to a long. If a non-numeric value is found, the
// return value is whatever has been accumulated so far. end_ptr always points
// to the character that caused the conversion to stop. We can't use strtol()
// etc because that would add a new dependency on libc. Eventually, this
// function could be made generally available within the library and used to
// replace the existing calls to strtol(). Currenly only handles non-negative
// numbers and does not detect overflow.
long
malloc_common_convert_to_long(const char * __null_terminated ptr, const char * __null_terminated *end_ptr)
{
#if !MALLOC_TARGET_EXCLAVES
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
#else
	return strtol(ptr, (char * __null_terminated *)end_ptr, 10);
#endif // MALLOC_TARGET_EXCLAVES
}

// Looks for a sequence of the form "key=value" in the string 'src' and
// returns the location of the first character of 'value', or NULL if not
// found. No spaces are permitted around the "=".
const char * __null_terminated
malloc_common_value_for_key(const char * __null_terminated src, const char * __null_terminated key)
{
	const char * __null_terminated ptr = src;
	size_t keylen = strlen(key);
	while ((ptr = malloc_common_strstr(ptr, __unsafe_forge_bidi_indexable(const char *, key, keylen), keylen)) != NULL) {
		// Workaround for indexable pointers being incrementable by one only
		for (size_t i = 0; i < keylen; ++i) {
			++ptr;
		}
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
const char * __null_terminated
malloc_common_value_for_key_copy(const char * __null_terminated src, const char * __null_terminated key,
							   char * __counted_by(maxlen) bufp, size_t maxlen)
{
	const char * __null_terminated ptr = malloc_common_value_for_key(src, key);
	if (ptr) {
		size_t to_len = maxlen;
		char * __counted_by(to_len) to = bufp;
		while (to_len > 1) { // Always leave room for a '\0'
			char c = *ptr++;
			if (c == '\0' || c == ' ' || c == '\t' || c == '\n') {
				break;
			}
			*to = c;
			++to;
			to_len--;
		}
		*to = '\0';	// Always nul-terminate
		return __unsafe_null_terminated_from_indexable(bufp, to);
	}
	return NULL;
}

unsigned
malloc_zone_batch_malloc_fallback(malloc_zone_t *zone, size_t size,
		void * __unsafe_indexable * __counted_by(num_requested)  results, unsigned num_requested)
{
	unsigned allocated;
	for (allocated = 0; allocated < num_requested; allocated++) {
		void *ptr = zone->malloc(zone, size);
		if (!ptr) {
			break;
		}

		results[allocated] = ptr;
	}

	return allocated;
}

void
malloc_zone_batch_free_fallback(malloc_zone_t *zone,
		void * __unsafe_indexable * __counted_by(count) to_be_freed,
		unsigned count)
{
	for (unsigned i = 1; i <= count; i++) {
		// Note: we iterate backward because nano and magazine malloc both do,
		// although that seems likely to just be a vestigial codegen
		// optimization for ancient non-optimizing compilers
		void * __unsafe_indexable ptr = to_be_freed[count - i];
		if (ptr) {
			zone->free(zone, ptr);
		}
	}
}

size_t
malloc_zone_pressure_relief_fallback(malloc_zone_t *zone, size_t goal)
{
	return 0;
}


MALLOC_NOEXPORT MALLOC_NOINLINE
void
___BUG_IN_CLIENT_OF_LIBMALLOC_POINTER_BEING_FREED_WAS_NOT_ALLOCATED(
		int flags, void *__unsafe_indexable ptr)
{

	malloc_report(flags, "*** error for object %p: "
		"pointer being freed was not allocated\n", ptr);
}

#if !MALLOC_TARGET_EXCLAVES && !MALLOC_TARGET_EXCLAVES_INTROSPECTOR

#if CONFIG_CHECK_PLATFORM_BINARY
// Avoid conditioning on this if at all possible
bool malloc_is_platform_binary = true;
#endif // CONFIG_CHECK_PLATFORM_BINARY

// Use malloc_is_platform_binary instead
bool
_malloc_is_platform_binary(void)
{
	uint32_t flags = 0;
	int err = csops(getpid(), CS_OPS_STATUS, &flags, sizeof(flags));
	if (err) {
		return false;
	}
	return (flags & CS_PLATFORM_BINARY);
}

bool malloc_internal_security_policy = false;

bool
_malloc_allow_internal_security_policy(const char *envp[])
{
#if !TARGET_OS_SIMULATOR && defined(_COMM_PAGE_DEV_FIRM)
	if (!*((uint32_t *)_COMM_PAGE_DEV_FIRM)) {
		return false;
	}
#endif

#if CONFIG_FEATUREFLAGS_SIMPLE
	if (os_feature_enabled_simple(libmalloc, AllowInternalSecurityPolicy,
			false)) {
		return true;
	}
#endif

	const char *flag = _simple_getenv(envp, "MallocAllowInternalSecurity");
	if (flag) {
		const char *endp;
		long value = malloc_common_convert_to_long(flag, &endp);
		if (!*endp && endp != flag && (value == 0 || value == 1)) {
			return (bool)value;
		}
	}

	return false;
}

#endif // !MALLOC_TARGET_EXCLAVES && !MALLOC_TARGET_EXCLAVES_INTROSPECTOR
