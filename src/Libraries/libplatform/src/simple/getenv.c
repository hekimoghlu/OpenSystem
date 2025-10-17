/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 12, 2023.
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
#include <TargetConditionals.h>

#include <stdlib.h>

#include <_simple.h>

#include <platform/string.h>
#include <platform/compat.h>

// This file is built with -fno-builtin to prevent the compiler from turning
// _simple_memcmp into a memcmp() call.

static int
_simple_memcmp(const void *s1, const void *s2, size_t n)
{
	if (n != 0) {
		const unsigned char *p1 = s1, *p2 = s2;

		do {
			if (*p1++ != *p2++)
				return (*--p1 - *--p2);
		} while (--n != 0);
	}
	return (0);
}

const char *
_simple_getenv(const char *envp[], const char *var) {
    const char **p;
    size_t var_len;

    var_len = strlen(var);

    for (p = envp; p && *p; p++) {
        size_t p_len = strlen(*p);

        if (p_len >= var_len &&
            _simple_memcmp(*p, var, var_len) == 0 &&
            (*p)[var_len] == '=') {
            return &(*p)[var_len + 1];
        }
    }

    return NULL;
}
