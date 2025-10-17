/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
#include <platform/string.h>

#if !_PLATFORM_OPTIMIZED_STRCPY

char *
_platform_strcpy(char * restrict dst, const char * restrict src) {
	const size_t length = _platform_strlen(src);
    //  The stpcpy() and strcpy() functions copy the string src to dst
    //  (including the terminating '\0' character).
    _platform_memmove(dst, src, length+1);
    //  The strcpy() and strncpy() functions return dst.
    return dst;
}

#if VARIANT_STATIC
char *
strcpy(char * restrict dst, const char * restrict src) {
	return _platform_strcpy(dst, src);
}
#endif

#endif
