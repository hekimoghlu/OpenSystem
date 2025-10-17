/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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
/****************************************************************************
 *  Author: Thomas E. Dickey                                                *
 ****************************************************************************/

/*
**	lib_mvcur.c
**/

#include <curses.priv.h>

MODULE_ID("$Id: strings.c,v 1.8 2012/02/22 22:34:31 tom Exp $")

/****************************************************************************
 * Useful string functions (especially for mvcur)
 ****************************************************************************/

#if !HAVE_STRSTR
NCURSES_EXPORT(char *)
_nc_strstr(const char *haystack, const char *needle)
{
    size_t len1 = strlen(haystack);
    size_t len2 = strlen(needle);
    char *result = 0;

    while ((len1 != 0) && (len1-- >= len2)) {
	if (!strncmp(haystack, needle, len2)) {
	    result = (char *) haystack;
	    break;
	}
	haystack++;
    }
    return result;
}
#endif

/*
 * Initialize the descriptor so we can append to it.  Note that 'src' may
 * be a null pointer (see _nc_str_null), so the corresponding strcat and
 * strcpy calls have to allow for this.
 */
NCURSES_EXPORT(string_desc *)
_nc_str_init(string_desc * dst, char *src, size_t len)
{
    if (dst != 0) {
	dst->s_head = src;
	dst->s_tail = src;
	dst->s_size = len - 1;
	dst->s_init = dst->s_size;
	if (src != 0)
	    *src = 0;
    }
    return dst;
}

/*
 * Initialize the descriptor for only tracking the amount of memory used.
 */
NCURSES_EXPORT(string_desc *)
_nc_str_null(string_desc * dst, size_t len)
{
    return _nc_str_init(dst, 0, len);
}

/*
 * Copy a descriptor
 */
NCURSES_EXPORT(string_desc *)
_nc_str_copy(string_desc * dst, string_desc * src)
{
    *dst = *src;
    return dst;
}

/*
 * Replaces strcat into a fixed buffer, returning false on failure.
 */
NCURSES_EXPORT(bool)
_nc_safe_strcat(string_desc * dst, const char *src)
{
    if (src != 0) {
	size_t len = strlen(src);

	if (len < dst->s_size) {
	    if (dst->s_tail != 0) {
		_nc_STRCPY(dst->s_tail, src, dst->s_size);
		dst->s_tail += len;
	    }
	    dst->s_size -= len;
	    return TRUE;
	}
    }
    return FALSE;
}

/*
 * Replaces strcpy into a fixed buffer, returning false on failure.
 */
NCURSES_EXPORT(bool)
_nc_safe_strcpy(string_desc * dst, const char *src)
{
    if (src != 0) {
	size_t len = strlen(src);

	if (len < dst->s_size) {
	    if (dst->s_head != 0) {
		_nc_STRCPY(dst->s_head, src, dst->s_size);
		dst->s_tail = dst->s_head + len;
	    }
	    dst->s_size = dst->s_init - len;
	    return TRUE;
	}
    }
    return FALSE;
}
